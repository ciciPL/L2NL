from __future__ import annotations
from urllib.parse import urlparse


def _is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".localhost")


class LLMClient:
    """OpenAI-compatible chat client. Online vs offline differ only by base_url."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 60, max_tokens: int = 512):
        from openai import OpenAI
        import httpx
        self.model = model
        self.max_tokens = max_tokens
        self.disable_thinking = _is_local_base_url(base_url)
        http_client = None
        if self.disable_thinking:
            http_client = httpx.Client(trust_env=False, timeout=timeout)
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-no-key",
            timeout=timeout,
            http_client=http_client,
        )

    def chat(self, messages: list[dict], temperature: float,
             stop: list[str] | None = None) -> str:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if self.disable_thinking else None
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            stop=stop, max_tokens=self.max_tokens, extra_body=extra_body,
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
