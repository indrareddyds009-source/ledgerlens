"""Append-only JSONL audit log of every turn (PII-redacted)."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

from .guardrails import redact

LOG_DIR = Path(os.getenv("LEDGERLENS_LOG_DIR", "logs"))


def log_turn(customer_id: str, thread_id: str, question: str, result: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "customer_id": customer_id,
           "thread_id": thread_id, "question": redact(question), "answer": redact(result["answer"]),
           "tool_calls": result["tool_calls"], "blocked": result["blocked"], "latency_ms": result["latency_ms"]}
    with open(LOG_DIR / f"{customer_id}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
