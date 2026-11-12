"""Input and output guardrails.

Input: block prompt-injection attempts and requests for other customers' data
before the model ever sees them. Output: redact PII (card numbers, SSNs,
emails, phone numbers) from anything returned to the user or written to logs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CARD = re.compile(r"\b(?:\d[ -]?){12,15}(\d{4})\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"(?<!\d)(?:\+1[ -]?)?\(?\d{3}\)?[ -.]\d{3}[ -.]\d{4}(?!\d)")

INJECTION = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"(reveal|show|print|repeat).{0,20}(system prompt|instructions)",
    r"you are now", r"developer mode", r"jailbreak",
    r"\b(drop|delete|truncate|update|insert)\s+(table|from|into)\b",
    r"\bselect\b.+\bfrom\b",
]
CROSS_CUSTOMER = [
    r"\bcust_\d+\b", r"(other|another|all) (customers?|users?|accounts?)",
    r"someone else'?s", r"\b(his|her|their) (account|transactions|spending)\b",
]


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


def redact(text: str) -> str:
    text = CARD.sub(lambda m: f"[CARD ••••{m.group(1)}]", text)
    text = SSN.sub("[SSN REDACTED]", text)
    text = EMAIL.sub("[EMAIL REDACTED]", text)
    return PHONE.sub("[PHONE REDACTED]", text)


def check_input(question: str, customer_id: str) -> GuardResult:
    q = question.lower()
    if len(question) > 2000:
        return GuardResult(False, "too_long")
    for pat in INJECTION:
        if re.search(pat, q):
            return GuardResult(False, "prompt_injection")
    for pat in CROSS_CUSTOMER:
        m = re.search(pat, q)
        if m and m.group(0) != customer_id:
            return GuardResult(False, "cross_customer")
    return GuardResult(True)


REFUSALS = {
    "prompt_injection": "I can't help with that. I can only answer questions about your own transactions.",
    "cross_customer": "I can only access your own account. I can't look up other customers' data.",
    "too_long": "That message is too long. Please ask a shorter question about your transactions.",
}
