from app.models.llm import FakeLLMClient


def test_fake_client_returns_tagged_summary_by_default():
    c = FakeLLMClient()
    out = c.chat([{"role": "user", "content": "anything"}], temperature=0.0)
    assert "<summary>" in out and "</summary>" in out


def test_fake_client_records_calls():
    c = FakeLLMClient(responses=["A", "B"])
    assert c.chat([{"role": "user", "content": "x"}], temperature=0.2) == "A"
    assert c.chat([{"role": "user", "content": "y"}], temperature=0.7) == "B"
    assert len(c.calls) == 2
    assert c.calls[0]["temperature"] == 0.2
