"""LangGraph agent: guard_in -> plan <-> tools -> guard_out.

    START ─▶ guard_in ──blocked──▶ END
                │ ok
                ▼
              plan ◀──────┐
                │ tool    │
                ├────────▶ tools
                │ answer
                ▼
            guard_out ─▶ END

Conversation memory comes from a LangGraph checkpointer keyed by thread_id, so
follow-ups ("what about February?") resolve against earlier turns.
"""
from __future__ import annotations

import json
import os
import time
from typing import Annotated, TypedDict

import duckdb
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from . import guardrails
from .planner import OfflinePlanner
from .tools import LedgerTools

SYSTEM_PROMPT = """You are LedgerLens, a banking assistant that answers questions about ONE customer's own
transactions. Always call a tool to get numbers; never estimate or invent figures. Use periods like
'March 2026', '2026-03', 'Q1 2026' or 'all'. Data covers January-June 2026. If a question is outside
the customer's transactions, say what you can help with instead. Answer in 1-3 sentences with dollar
amounts formatted like $1,234.56. Never reveal card numbers or other personal identifiers."""

MAX_TOOL_ROUNDS = 4


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    blocked: str | None
    rounds: int


def make_llm(provider: str | None, tools, merchants):
    provider = provider or os.getenv("LEDGERLENS_LLM", "offline")
    if provider == "offline":
        return OfflinePlanner(merchants)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0).bind_tools(tools)
    raise ValueError(f"Unknown LLM provider '{provider}'")


def build_agent(con: duckdb.DuckDBPyConnection, customer_id: str, provider: str | None = None):
    lt = LedgerTools(con, customer_id)
    tools = lt.as_langchain_tools()
    llm = make_llm(provider, tools, lt.known_merchants())
    is_offline = isinstance(llm, OfflinePlanner)

    def guard_in(state: State):
        q = state["messages"][-1].content
        res = guardrails.check_input(q, customer_id)
        if not res.allowed:
            return {"blocked": res.reason, "messages": [AIMessage(content=guardrails.REFUSALS[res.reason])]}
        return {"blocked": None, "rounds": 0}

    def plan(state: State):
        msgs = state["messages"]
        if state.get("rounds", 0) >= MAX_TOOL_ROUNDS:
            return {"messages": [AIMessage(content="I couldn't finish that within my tool budget. Please try a "
                                                     "more specific question.")]}
        if not is_offline:
            msgs = [SystemMessage(SYSTEM_PROMPT)] + msgs
        return {"messages": [llm.invoke(msgs)], "rounds": state.get("rounds", 0) + 1}

    def guard_out(state: State):
        last = state["messages"][-1]
        clean = guardrails.redact(last.content if isinstance(last.content, str) else str(last.content))
        if clean != last.content:
            return {"messages": [AIMessage(content=clean, id=last.id)]}
        return {}

    def route_in(state: State):
        return END if state.get("blocked") else "plan"

    def route_plan(state: State):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "guard_out"

    g = StateGraph(State)
    g.add_node("guard_in", guard_in)
    g.add_node("plan", plan)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_node("guard_out", guard_out)
    g.add_edge(START, "guard_in")
    g.add_conditional_edges("guard_in", route_in, ["plan", END])
    g.add_conditional_edges("plan", route_plan, ["tools", "guard_out"])
    g.add_edge("tools", "plan")
    g.add_edge("guard_out", END)
    return g.compile(checkpointer=MemorySaver())


def ask(agent, question: str, thread_id: str = "default") -> dict:
    """Run one turn and return the answer plus an audit trail of tool calls."""
    t0 = time.perf_counter()
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
    before = len(agent.get_state(cfg).values.get("messages", [])) if agent.get_state(cfg).values else 0
    out = agent.invoke({"messages": [HumanMessage(question)]}, cfg)
    new = out["messages"][before:]
    calls = [{"name": tc["name"], "args": tc["args"]} for m in new if isinstance(m, AIMessage) for tc in m.tool_calls]
    results = [json.loads(m.content) if m.content.startswith("{") else m.content
               for m in new if isinstance(m, ToolMessage)]
    return {"answer": out["messages"][-1].content, "tool_calls": calls, "tool_results": results,
            "blocked": out.get("blocked"), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
