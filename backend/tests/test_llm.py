from app.models.llm import FakeLLMClient
from app.models.llm import LLMClient
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import json


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


def test_llm_client_bypasses_proxy_for_localhost(monkeypatch):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0") or 0)
            self.rfile.read(n)
            body = json.dumps({"choices": [{"message": {"content": "LOCAL_OK"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    try:
        c = LLMClient(
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            api_key="sk-test",
            model="fake",
            timeout=2,
            max_tokens=8,
        )
        assert c.chat([{"role": "user", "content": "ping"}], temperature=0.0) == "LOCAL_OK"
    finally:
        srv.shutdown()


def test_llm_client_disables_thinking_for_localhost_llama_cpp():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0") or 0)
            seen["body"] = json.loads(self.rfile.read(n))
            body = json.dumps({"choices": [{"message": {"content": "NO_THINK_OK"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        c = LLMClient(
            base_url=f"http://127.0.0.1:{srv.server_port}/v1",
            api_key="sk-test",
            model="fake",
            timeout=2,
            max_tokens=8,
        )
        assert c.chat([{"role": "user", "content": "ping"}], temperature=0.0) == "NO_THINK_OK"
        assert seen["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        srv.shutdown()
