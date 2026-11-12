from fastapi.testclient import TestClient

from ledgerlens.api import app

client = TestClient(app)
H = {"Authorization": "Bearer demo-alex"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_requires_token():
    assert client.post("/ask", json={"question": "hi"}).status_code == 401


def test_ask_and_follow_up(tmp_path, monkeypatch):
    monkeypatch.setattr("ledgerlens.audit.LOG_DIR", tmp_path)
    r = client.post("/ask", json={"question": "How much did I spend in March?"}, headers=H).json()
    assert "$" in r["answer"] and r["tool_calls"][0]["name"] == "spending_summary"
    r2 = client.post("/ask", json={"question": "what about May?", "thread_id": r["thread_id"]}, headers=H).json()
    assert "May 2026" in r2["answer"]
    assert (tmp_path / "cust_001.jsonl").read_text().count("\n") == 2


def test_thread_ids_are_namespaced_per_customer(tmp_path, monkeypatch):
    monkeypatch.setattr("ledgerlens.audit.LOG_DIR", tmp_path)
    r = client.post("/ask", json={"question": "Top 3 merchants in March"}, headers=H).json()
    other = {"Authorization": "Bearer demo-priya"}
    r2 = client.post("/ask", json={"question": "what about April?", "thread_id": r["thread_id"]}, headers=other).json()
    assert r2["tool_calls"] == []  # Priya cannot continue Alex's conversation
