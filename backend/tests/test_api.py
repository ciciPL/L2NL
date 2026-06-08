from fastapi.testclient import TestClient
import app.main as main
from app.models.llm import FakeLLMClient


def _client(monkeypatch):
    # Force the app to use a deterministic LLM regardless of request model config.
    monkeypatch.setattr(main, "make_llm",
                        lambda cfg: FakeLLMClient(responses=["It runs."]))
    return TestClient(main.app)


def test_health(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summarize_returns_summary_and_trace(monkeypatch):
    c = _client(monkeypatch)
    body = {
        "code": "def foo; 1; end",
        "language": "ruby",
        "model": {"base_url": "http://x/v1", "api_key": "k", "model": "m"},
        "trace": True,
    }
    r = c.post("/summarize", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "It runs."
    assert data["trace"]["translation"]["pivot_code"] == "def foo; 1; end"


def test_summarize_validation_error_returns_422(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/summarize", json={"language": "ruby"})  # missing code+model
    assert r.status_code == 422
