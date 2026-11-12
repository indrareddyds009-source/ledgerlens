"""Customer-scoped analysis tools the agent is allowed to call.

Design rules:
* The LLM never writes SQL. It only picks a tool and fills typed arguments.
* Every query is parameterized and filtered by the authenticated customer_id,
  which is bound when the tools are created, never taken from the model.
* Tools return plain dicts so results are easy to log, test and evaluate.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re

import duckdb
from langchain_core.tools import StructuredTool

from .guardrails import redact

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
DEFAULT_YEAR = 2026


def resolve_period(period: str | None) -> tuple[dt.date | None, dt.date | None, str]:
    """Turn 'March 2026', '2026-03', 'Q1 2026', 'march' or 'all' into [start, end) dates and a label."""
    p = (period or "all").strip().lower()
    if p in ("", "all", "all time", "overall", "total"):
        return None, None, "all time"
    if m := re.fullmatch(r"(\d{4})-(\d{2})", p):
        y, mo = int(m[1]), int(m[2])
        return _month(y, mo)
    if m := re.fullmatch(r"q([1-4])(?:\s+(\d{4}))?", p):
        q, y = int(m[1]), int(m[2] or DEFAULT_YEAR)
        start = dt.date(y, 3 * q - 2, 1)
        end = dt.date(y + (q == 4), (3 * q) % 12 + 1, 1)
        return start, end, f"Q{q} {y}"
    if (m := re.fullmatch(r"([a-z]+)\.?(?:\s+(\d{4}))?", p)) and m[1] in MONTHS:
        return _month(int(m[2] or DEFAULT_YEAR), MONTHS[m[1]])
    raise ValueError(f"Unrecognized period '{period}'. Use e.g. 'March 2026', '2026-03', 'Q1 2026' or 'all'.")


def _month(y: int, mo: int):
    start = dt.date(y, mo, 1)
    end = dt.date(y + (mo == 12), mo % 12 + 1, 1)
    return start, end, f"{calendar.month_name[mo]} {y}"


class LedgerTools:
    def __init__(self, con: duckdb.DuckDBPyConnection, customer_id: str):
        self.con = con
        self.customer_id = customer_id

    # ---- helpers ---------------------------------------------------------
    def _where(self, period=None, category=None, extra=""):
        start, end, label = resolve_period(period)
        clauses, params = ["customer_id = ?"], [self.customer_id]
        if start:
            clauses += ["txn_date >= ?", "txn_date < ?"]
            params += [start, end]
        if category:
            clauses.append("lower(category) = lower(?)")
            params.append(category)
        if extra:
            clauses.append(extra)
        return " AND ".join(clauses), params, label

    def _q(self, sql, params):
        return self.con.execute(sql, params).fetchall()

    # ---- tools -----------------------------------------------------------
    def spending_summary(self, period: str = "all", category: str | None = None) -> dict:
        """Total spending (debits), income (credits), net cash flow and a per-category breakdown for a period."""
        where, params, label = self._where(period, category)
        debit, credit, n = self._q(
            f"SELECT COALESCE(SUM(amount) FILTER (WHERE direction='debit'),0),"
            f" COALESCE(SUM(amount) FILTER (WHERE direction='credit'),0), COUNT(*) FROM transactions WHERE {where}",
            params)[0]
        cats = self._q(f"SELECT category, SUM(amount) s FROM transactions WHERE {where} AND direction='debit'"
                       f" GROUP BY 1 ORDER BY s DESC", params)
        return {"period": label, "category": category, "total_spent": round(float(debit), 2),
                "total_income": round(float(credit), 2), "net": round(float(credit - debit), 2),
                "transactions": n, "by_category": {c: round(float(s), 2) for c, s in cats}}

    def top_merchants(self, n: int = 5, period: str = "all", category: str | None = None) -> dict:
        """The N merchants where the customer spent the most in a period."""
        n = max(1, min(int(n), 20))
        where, params, label = self._where(period, category, "direction='debit'")
        rows = self._q(f"SELECT merchant, SUM(amount) s, COUNT(*) c FROM transactions WHERE {where}"
                       f" GROUP BY 1 ORDER BY s DESC LIMIT {n}", params)
        return {"period": label, "category": category,
                "merchants": [{"merchant": m, "spent": round(float(s), 2), "visits": c} for m, s, c in rows]}

    def monthly_trend(self, category: str | None = None) -> dict:
        """Month-by-month spending, optionally for one category, with the highest and lowest month."""
        where, params, _ = self._where(None, category, "direction='debit'")
        rows = self._q(f"SELECT strftime(txn_date, '%Y-%m') m, SUM(amount) FROM transactions WHERE {where}"
                       f" GROUP BY 1 ORDER BY 1", params)
        months = [{"month": m, "spent": round(float(s), 2)} for m, s in rows]
        top = max(months, key=lambda r: r["spent"]) if months else None
        low = min(months, key=lambda r: r["spent"]) if months else None
        return {"category": category, "months": months, "highest": top, "lowest": low}

    def compare_periods(self, period_a: str, period_b: str) -> dict:
        """Compare spending between two periods, overall and by category."""
        a, b = self.spending_summary(period_a), self.spending_summary(period_b)
        cats = sorted(set(a["by_category"]) | set(b["by_category"]))
        deltas = {c: round(b["by_category"].get(c, 0) - a["by_category"].get(c, 0), 2) for c in cats}
        biggest = max(deltas.items(), key=lambda kv: abs(kv[1])) if deltas else None
        return {"period_a": a["period"], "period_b": b["period"], "spent_a": a["total_spent"],
                "spent_b": b["total_spent"], "change": round(b["total_spent"] - a["total_spent"], 2),
                "category_change": deltas, "biggest_mover": biggest}

    def search_transactions(self, merchant: str | None = None, min_amount: float | None = None,
                            period: str = "all", limit: int = 10) -> dict:
        """List individual transactions, filtered by merchant name and/or minimum amount."""
        extra, extra_params = [], []
        if merchant:
            extra.append("lower(merchant) LIKE ?")
            extra_params.append(f"%{merchant.lower()}%")
        if min_amount is not None:
            extra.append("amount >= ?")
            extra_params.append(float(min_amount))
        where, params, label = self._where(period, None, " AND ".join(extra))
        params += extra_params
        limit = max(1, min(int(limit), 50))
        rows = self._q(f"SELECT txn_date, merchant, category, direction, amount, description FROM transactions"
                       f" WHERE {where} ORDER BY amount DESC, txn_date LIMIT {limit}", params)
        total = self._q(f"SELECT COUNT(*), COALESCE(SUM(amount),0) FROM transactions WHERE {where}", params)[0]
        return {"period": label, "merchant": merchant, "min_amount": min_amount, "count": total[0],
                "total": round(float(total[1]), 2),
                "transactions": [{"date": str(d), "merchant": m, "category": c, "direction": dr,
                                  "amount": round(float(a), 2), "memo": redact(memo)} for d, m, c, dr, a, memo in rows]}

    def detect_anomalies(self, z_threshold: float = 3.0) -> dict:
        """Flag debits that are unusually large for their category (z-score above threshold)."""
        rows = self._q("""
            WITH t AS (
              SELECT *, AVG(amount) OVER (PARTITION BY category) mu,
                        STDDEV_SAMP(amount) OVER (PARTITION BY category) sd
              FROM transactions WHERE customer_id = ? AND direction = 'debit')
            SELECT txn_date, merchant, category, amount, (amount - mu) / NULLIF(sd, 0) z
            FROM t WHERE (amount - mu) / NULLIF(sd, 0) >= ? ORDER BY z DESC LIMIT 10""",
                       [self.customer_id, float(z_threshold)])
        return {"z_threshold": z_threshold,
                "anomalies": [{"date": str(d), "merchant": m, "category": c, "amount": round(float(a), 2),
                               "z_score": round(float(z), 1)} for d, m, c, a, z in rows]}

    def known_merchants(self) -> list[str]:
        return [r[0] for r in self._q("SELECT DISTINCT merchant FROM transactions WHERE customer_id = ?",
                                      [self.customer_id])]

    # ---- LangChain wiring -----------------------------------------------
    TOOL_NAMES = ("spending_summary", "top_merchants", "monthly_trend", "compare_periods",
                  "search_transactions", "detect_anomalies")

    def as_langchain_tools(self) -> list[StructuredTool]:
        return [StructuredTool.from_function(getattr(self, name), name=name) for name in self.TOOL_NAMES]
