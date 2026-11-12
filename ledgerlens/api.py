"""FastAPI service. The customer is identified by a bearer token, never by the request body."""
from __future__ import annotations

import os
import uuid
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .agent import ask, build_agent
from .audit import log_turn
from .data import build_ledger

# Demo tokens only. In production this is replaced by OAuth/JWT validation.
DEMO_TOKENS = {"demo-alex": "cust_001", "demo-priya": "cust_002", "demo-jordan": "cust_003"}

app = FastAPI(title="LedgerLens", version="0.1.0",
              description="Agentic, guard-railed Q&A over a customer's own transactions.")


@lru_cache
def _con():
    return build_ledger(os.getenv("LEDGERLENS_DB", ":memory:"))


@lru_cache
def _agent(customer_id: str):
    return build_agent(_con(), customer_id)


def current_customer(authorization: str = Header(default="")) -> str:
    token = authorization.removeprefix("Bearer ").strip()
    if token not in DEMO_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return DEMO_TOKENS[token]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    thread_id: str
    tool_calls: list[dict]
    blocked: str | None
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "llm": os.getenv("LEDGERLENS_LLM", "offline")}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, customer_id: str = Depends(current_customer)):
    thread_id = req.thread_id or uuid.uuid4().hex
    # Namespace threads by customer so one user can never resume another's conversation.
    result = ask(_agent(customer_id), req.question, thread_id=f"{customer_id}:{thread_id}")
    log_turn(customer_id, thread_id, req.question, result)
    return AskResponse(answer=result["answer"], thread_id=thread_id, tool_calls=result["tool_calls"],
                       blocked=result["blocked"], latency_ms=result["latency_ms"])
