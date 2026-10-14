from fastapi.testclient import TestClient

import api.main
from api.main import app
from services.llm import LLMError

client = TestClient(app)


def test_chat_success(fake_llm):
    res = client.post("/chat", json={"question": "What is RAG?"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"].startswith("RAG")
    assert body["prompt_tokens"] == 5
    assert body["completion_tokens"] == 7


def test_chat_empty_question_returns_422():
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_chat_groq_failure_returns_502(monkeypatch):
    async def failing_ask_llm(question):
        raise LLMError("Groq request failed: service unavailable")

    monkeypatch.setattr(api.main, "ask_llm", failing_ask_llm)
    res = client.post("/chat", json={"question": "Hello?"})

    assert res.status_code == 502
    assert "Groq request failed" in res.json()["detail"]
