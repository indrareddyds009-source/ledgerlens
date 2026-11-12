import datetime as dt

import pytest

from ledgerlens.tools import LedgerTools, resolve_period


@pytest.mark.parametrize("p,label", [("March 2026", "March 2026"), ("2026-03", "March 2026"), ("mar", "March 2026"),
                                     ("Q2 2026", "Q2 2026"), ("q4", "Q4 2026"), ("all", "all time")])
def test_resolve_period(p, label):
    assert resolve_period(p)[2] == label


def test_q4_end_rolls_year():
    assert resolve_period("Q4 2026")[1] == dt.date(2027, 1, 1)


def test_bad_period():
    with pytest.raises(ValueError):
        resolve_period("someday")


def test_summary_matches_raw_sql(con):
    r = LedgerTools(con, "cust_001").spending_summary("2026-02")
    raw = con.execute("SELECT SUM(amount) FROM transactions WHERE customer_id='cust_001' AND direction='debit'"
                      " AND txn_date >= '2026-02-01' AND txn_date < '2026-03-01'").fetchone()[0]
    assert r["total_spent"] == round(float(raw), 2)
    assert abs(sum(r["by_category"].values()) - r["total_spent"]) < 0.05


def test_customer_scoping(con):
    """Tools for one customer must never see another customer's rows."""
    a = LedgerTools(con, "cust_001").search_transactions(limit=50)
    b = LedgerTools(con, "cust_002").search_transactions(limit=50)
    assert a["count"] != b["count"] or a["total"] != b["total"]
    own = con.execute("SELECT COUNT(*) FROM transactions WHERE customer_id='cust_001'").fetchone()[0]
    assert a["count"] == own


def test_sql_injection_in_args_is_inert(con):
    r = LedgerTools(con, "cust_001").search_transactions(merchant="x' OR '1'='1")
    assert r["count"] == 0


def test_memos_are_redacted(con):
    r = LedgerTools(con, "cust_001").search_transactions(merchant="Best Buy", limit=50)
    assert all("4111 1111" not in t["memo"] for t in r["transactions"])


def test_anomaly_detected(con):
    r = LedgerTools(con, "cust_001").detect_anomalies()
    assert r["anomalies"] and r["anomalies"][0]["merchant"] == "Best Buy"
