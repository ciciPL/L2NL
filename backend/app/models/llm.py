from __future__ import annotations


class LLMClient:
    """OpenAI-compatible chat client. Online vs offline differ only by base_url."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key or "sk-no-key")

    def chat(self, messages: list[dict], temperature: float,
             stop: list[str] | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            stop=stop,
        )
        return resp.choices[0].message.content or ""


class FakeLLMClient(LLMClient):
    """Deterministic client for tests/stubs. No network."""

    def __init__(self, responses: list[str] | None = None):
        self.model = "fake"
        self._responses = list(responses) if responses else None
        self._i = 0
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], temperature: float,
             stop: list[str] | None = None) -> str:
        self.calls.append({"messages": messages, "temperature": temperature,
                           "stop": stop})
        if self._responses is not None:
            out = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
            return out
        return "<summary>Stubbed summary of the provided code.</summary>"
