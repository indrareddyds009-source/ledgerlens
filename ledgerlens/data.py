"""Synthetic transaction ledger, stored in DuckDB.

Generates realistic, reproducible card/bank activity for a handful of demo customers:
monthly salary credits, recurring bills, everyday spend, and a few injected anomalies.
No real customer data is used anywhere in this project.
"""
from __future__ import annotations

import datetime as dt
import random

import duckdb

CATEGORIES: dict[str, list[tuple[str, float, float]]] = {
    # category: [(merchant, min_amount, max_amount)]
    "Groceries": [("Whole Foods", 25, 180), ("Safeway", 15, 140), ("Trader Joe's", 20, 120)],
    "Dining": [("Chipotle", 9, 22), ("Starbucks", 4, 12), ("Sweetgreen", 12, 20), ("Olive Garden", 30, 95)],
    "Transport": [("Uber", 8, 45), ("Lyft", 8, 40), ("Shell", 35, 75)],
    "Utilities": [("PG&E", 80, 190), ("Comcast", 70, 90), ("T-Mobile", 65, 85)],
    "Shopping": [("Target", 15, 160), ("Amazon Marketplace", 10, 220), ("Best Buy", 25, 400)],
    "Travel": [("United Airlines", 180, 650), ("Marriott", 150, 420)],
    "Health": [("CVS Pharmacy", 8, 60), ("Walgreens", 6, 55)],
    "Entertainment": [("Netflix", 15.49, 15.49), ("Spotify", 11.99, 11.99), ("AMC Theatres", 14, 48)],
}
RECURRING = {"Netflix", "Spotify", "Comcast", "T-Mobile", "PG&E"}

CUSTOMERS = {
    "cust_001": {"name": "Alex Rivera", "salary": 7200.00},
    "cust_002": {"name": "Priya Shah", "salary": 9100.00},
    "cust_003": {"name": "Jordan Lee", "salary": 5400.00},
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR PRIMARY KEY,
    name        VARCHAR
);
CREATE TABLE IF NOT EXISTS transactions (
    txn_id      VARCHAR PRIMARY KEY,
    customer_id VARCHAR,
    txn_date    DATE,
    merchant    VARCHAR,
    category    VARCHAR,
    direction   VARCHAR,          -- 'debit' | 'credit'
    amount      DECIMAL(12, 2),   -- always positive
    channel     VARCHAR,          -- 'card' | 'ach' | 'online'
    description VARCHAR           -- raw memo; may contain PII that must never reach the user unredacted
);
"""


def _month_starts(start: dt.date, months: int) -> list[dt.date]:
    out, y, m = [], start.year, start.month
    for _ in range(months):
        out.append(dt.date(y, m, 1))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def generate_rows(seed: int = 7, start: dt.date = dt.date(2026, 1, 1), months: int = 6) -> list[tuple]:
    rng = random.Random(seed)
    rows: list[tuple] = []
    n = 0
    for cid, profile in CUSTOMERS.items():
        card_last4 = f"{rng.randint(1000, 9999)}"
        fake_card = f"4111 1111 1111 {card_last4}"  # Visa test-number prefix: synthetic PII
        for ms in _month_starts(start, months):
            nxt = dt.date(ms.year + (ms.month == 12), ms.month % 12 + 1, 1)
            days = (nxt - ms).days
            # salary on the 1st and 15th
            for d in (1, 15):
                n += 1
                rows.append((f"T{n:06d}", cid, ms.replace(day=d), "Employer Payroll", "Income", "credit",
                             round(profile["salary"] / 2, 2), "ach", f"PAYROLL DEP ACCT ****{card_last4}"))
            for cat, merchants in CATEGORIES.items():
                for merchant, lo, hi in merchants:
                    count = 1 if merchant in RECURRING else rng.randint(0, 6 if cat in ("Dining", "Groceries", "Transport") else 2)
                    if cat == "Travel":
                        count = 1 if rng.random() < 0.25 else 0
                    for _ in range(count):
                        n += 1
                        day = 3 if merchant in RECURRING else rng.randint(1, days)
                        amt = lo if lo == hi else round(rng.uniform(lo, hi), 2)
                        memo = f"POS {merchant.upper()} CARD {fake_card}" if rng.random() < 0.15 else f"POS {merchant.upper()}"
                        rows.append((f"T{n:06d}", cid, ms.replace(day=day), merchant, cat, "debit", amt,
                                     "online" if "Amazon" in merchant or merchant in RECURRING else "card", memo))
        # one injected anomaly per customer: an unusually large electronics purchase
        n += 1
        anomaly_month = _month_starts(start, months)[rng.randint(2, months - 1)]
        rows.append((f"T{n:06d}", cid, anomaly_month.replace(day=rng.randint(5, 25)), "Best Buy", "Shopping",
                     "debit", round(rng.uniform(2400, 3200), 2), "card", f"POS BEST BUY CARD {fake_card}"))
    return rows


def build_ledger(path: str = ":memory:", seed: int = 7, months: int = 6) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    con.execute(SCHEMA)
    if con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0:
        con.executemany("INSERT INTO customers VALUES (?, ?)", [(k, v["name"]) for k, v in CUSTOMERS.items()])
        con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        generate_rows(seed=seed, months=months))
    return con


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "ledger.duckdb"
    c = build_ledger(out)
    print(c.execute("SELECT customer_id, COUNT(*), MIN(txn_date), MAX(txn_date) FROM transactions GROUP BY 1 ORDER BY 1").fetchall())
