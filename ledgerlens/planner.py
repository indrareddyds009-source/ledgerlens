"""Deterministic offline planner and answer writer.

Lets the full LangGraph pipeline (guardrails -> plan -> tools -> answer) run with no
API key: in CI, in the eval harness's baseline, and for local demos. It emits the
same tool-call messages an LLM would, so the graph code path is identical.
"""
from __future__ import annotations

import re
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .tools import MONTHS

CATS = ["groceries", "dining", "transport", "utilities", "shopping", "travel", "health", "entertainment", "income"]
CAT_ALIASES = {"food": "Dining", "restaurants": "Dining", "eating out": "Dining", "gas": "Transport",
               "rides": "Transport", "bills": "Utilities", "flights": "Travel", "hotels": "Travel"}
PERIOD_RE = re.compile(r"\b(q[1-4](?:\s+\d{{4}})?|\d{{4}}-\d{{2}}|(?:{})(?:\s+\d{{4}})?)\b".format("|".join(
    sorted((k for k in MONTHS if len(k) > 2), key=len, reverse=True))))


def money(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _periods(q: str) -> list[str]:
    return [m.group(1) for m in PERIOD_RE.finditer(q)]


def _category(q: str):
    for c in CATS:
        if re.search(rf"\b{c}\b", q):
            return c.capitalize()
    for alias, c in CAT_ALIASES.items():
        if re.search(rf"\b{alias}\b", q):
            return c
    return None


def plan(question: str, history: list[BaseMessage], merchants: list[str]) -> tuple[str, dict] | None:
    q = question.lower()
    periods, cat = _periods(q), _category(q)
    period = periods[0] if periods else "all"
    if len(periods) >= 2 and re.search(r"compar|vs\.?|versus|differ|change", q):
        return "compare_periods", {"period_a": periods[0], "period_b": periods[1]}
    if re.search(r"unusual|anomal|suspicious|fraud|odd|strange", q):
        return "detect_anomalies", {}
    if re.search(r"trend|each month|by month|per month|month[- ]over[- ]month|which month", q):
        return "monthly_trend", {"category": cat}
    if m := re.search(r"top\s*(\d+)?\s*(merchants|stores|places|vendors)|where (do|did) i (spend|shop)", q):
        n = int(m.group(1)) if m.group(1) else 5
        return "top_merchants", {"n": n, "period": period, "category": cat}
    merchant = next((mm for mm in sorted(merchants, key=len, reverse=True) if mm.lower() in q), None)
    amt = re.search(r"(?:over|above|more than|>)\s*\$?([\d,]+(?:\.\d+)?)", q)
    if merchant or amt or re.search(r"\b(list|show)\b.*transactions", q):
        return "search_transactions", {"merchant": merchant, "period": period,
                                       "min_amount": float(amt.group(1).replace(",", "")) if amt else None}
    if re.search(r"spen[dt]|earn|income|net|cash ?flow|cost|paid|how much", q):
        return "spending_summary", {"period": period, "category": cat}
    # Follow-up such as "what about February?": reuse the previous tool with the new period.
    if periods:
        for msg in reversed(history):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                prev = msg.tool_calls[0]
                args = dict(prev["args"])
                if "period" in args:
                    args["period"] = periods[0]
                    return prev["name"], args
                if prev["name"] == "compare_periods":
                    return "spending_summary", {"period": periods[0]}
    return None


def write_answer(tool: str, r: dict) -> str:
    if tool == "spending_summary":
        scope = f" on {r['category']}" if r["category"] else ""
        s = f"For {r['period']}, you spent {money(r['total_spent'])}{scope}"
        if not r["category"]:
            s += f" and received {money(r['total_income'])}, for a net cash flow of {money(r['net'])}"
            if r["by_category"]:
                top = list(r["by_category"].items())[:3]
                s += ". Biggest categories: " + ", ".join(f"{c} {money(v)}" for c, v in top)
        return s + f" ({r['transactions']} transactions)."
    if tool == "top_merchants":
        if not r["merchants"]:
            return f"No spending found for {r['period']}."
        items = "; ".join(f"{i+1}. {m['merchant']} {money(m['spent'])} ({m['visits']} visits)"
                          for i, m in enumerate(r["merchants"]))
        return f"Top merchants for {r['period']}: {items}."
    if tool == "monthly_trend":
        if not r["months"]:
            return "No spending found."
        scope = f"{r['category']} spending" if r["category"] else "Spending"
        body = ", ".join(f"{m['month']} {money(m['spent'])}" for m in r["months"])
        return (f"{scope} by month: {body}. Highest was {r['highest']['month']} at {money(r['highest']['spent'])}; "
                f"lowest was {r['lowest']['month']} at {money(r['lowest']['spent'])}.")
    if tool == "compare_periods":
        direction = "more" if r["change"] >= 0 else "less"
        s = (f"You spent {money(r['spent_b'])} in {r['period_b']} vs {money(r['spent_a'])} in {r['period_a']}: "
             f"{money(abs(r['change']))} {direction}.")
        if r["biggest_mover"]:
            c, d = r["biggest_mover"]
            s += f" The biggest change was {c} ({'+' if d >= 0 else '-'}{money(abs(d))})."
        return s
    if tool == "search_transactions":
        if not r["count"]:
            return "No matching transactions found."
        rows = "; ".join(f"{t['date']} {t['merchant']} {money(t['amount'])}" for t in r["transactions"][:5])
        return f"Found {r['count']} matching transactions totaling {money(r['total'])} ({r['period']}). Largest: {rows}."
    if tool == "detect_anomalies":
        if not r["anomalies"]:
            return "Nothing looks unusual: no transactions stand out for their category."
        rows = "; ".join(f"{a['date']} {a['merchant']} {money(a['amount'])} ({a['category']}, z={a['z_score']})"
                         for a in r["anomalies"])
        return f"These transactions are unusually large for their category: {rows}. Please review them."
    return str(r)


class OfflinePlanner:
    """Duck-types the bits of a chat model the graph uses: .invoke(messages) -> AIMessage."""

    def __init__(self, merchants: list[str]):
        self.merchants = merchants

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            import json
            return AIMessage(content=write_answer(last.name, json.loads(last.content)))
        question = last.content if isinstance(last, HumanMessage) else ""
        choice = plan(question, messages[:-1], self.merchants)
        if choice is None:
            return AIMessage(content="I can answer questions about your spending, income, merchants, monthly "
                                     "trends, period comparisons and unusual transactions. Try: "
                                     "\"How much did I spend on dining in March?\"")
        name, args = choice
        args = {k: v for k, v in args.items() if v is not None}
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])
