from fastapi.testclient import TestClient
import hashlib
import json

import app.main as main
from app.models.llm import FakeLLMClient
from app.components.retriever import Retriever
from app.components.extractor import Extractor


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _client(monkeypatch):
    # Force the app to use a deterministic LLM regardless of request model config.
    monkeypatch.setattr(main, "make_llm",
                        lambda cfg: FakeLLMClient(
                            responses=["<PYTHON>\ndef foo():\n    pass\n</PYTHON>", "It runs."]))
    monkeypatch.setattr(main.settings, "allow_stubs", True)
    monkeypatch.setattr(main, "retriever", Retriever(allow_stubs=True))
    monkeypatch.setattr(main, "extractor", Extractor(allow_stubs=True))
    return TestClient(main.app)


def test_health(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "ready" in r.json()


def test_health_reports_missing_assets_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(main.settings, "extractor_weights", str(tmp_path / "missing.bin"))
    monkeypatch.setattr(main.settings, "corpus_path", str(tmp_path / "missing.jsonl"))
    monkeypatch.setattr(main.settings, "codebert_path", str(tmp_path / "missing-codebert"))
    monkeypatch.setattr(main.settings, "allow_stubs", False)
    c = TestClient(main.app)
    data = c.get("/health").json()
    assert data["ready"] is False
    assert data["assets"]["extractor"]["present"] is False
    assert data["assets"]["corpus"]["present"] is False
    assert data["assets"]["codebert"]["present"] is False


def test_health_reports_manifest_checksums(monkeypatch, tmp_path):
    extractor = tmp_path / "extractor" / "pytorch_model.bin"
    corpus = tmp_path / "corpus" / "corpus_30k.jsonl"
    codebert_dir = tmp_path / "codebert-base"
    codebert_archive = tmp_path / "codebert-base.tar.gz"
    extractor.parent.mkdir()
    corpus.parent.mkdir()
    codebert_dir.mkdir()
    (codebert_dir / "config.json").write_text("{}")
    extractor.write_bytes(b"extractor")
    corpus.write_bytes(b"corpus")
    codebert_archive.write_bytes(b"archive")
    manifest = tmp_path / "assets-manifest.v2.json"
    manifest.write_text(json.dumps({
        "version": 2,
        "release": "test",
        "sources": [{"id": "gitee", "baseUrl": "https://example.invalid", "enabledByDefault": True}],
        "assets": [
            {
                "name": "extractor/pytorch_model.bin",
                "target": "extractor/pytorch_model.bin",
                "sha256": _sha256(b"extractor"),
                "size": len(b"extractor"),
            },
            {
                "name": "corpus/corpus_30k.jsonl",
                "target": "corpus/corpus_30k.jsonl",
                "sha256": _sha256(b"corpus"),
                "size": len(b"corpus"),
            },
            {
                "name": "codebert-base",
                "target": "codebert-base.tar.gz",
                "extractTo": "codebert-base",
                "sha256": _sha256(b"archive"),
                "size": len(b"archive"),
            },
        ],
    }))
    monkeypatch.setattr(main.settings, "extractor_weights", str(extractor))
    monkeypatch.setattr(main.settings, "corpus_path", str(corpus))
    monkeypatch.setattr(main.settings, "codebert_path", str(codebert_dir))
    monkeypatch.setattr(main.settings, "assets_manifest", str(manifest))
    monkeypatch.setattr(main.settings, "allow_stubs", False)

    data = TestClient(main.app).get("/health").json()
    assert data["ready"] is True
    assert data["assets"]["extractor"]["checksum"] is True
    assert data["assets"]["corpus"]["checksum"] is True
    assert data["assets"]["codebert"]["checksum"] is True

    corpus.write_bytes(b"tampered")
    data = TestClient(main.app).get("/health").json()
    assert data["ready"] is False
    assert data["assets"]["corpus"]["checksum"] is False


def test_summarize_returns_summary_and_trace(monkeypatch):
    c = _client(monkeypatch)
    body = {
        "code": "def foo; 1; end",
        "language": "ruby",
        "model": {"base_url": "http://x/v1", "api_key": "k", "model": "m"},
        "params": {"temperatures": [0.0]},
        "trace": True,
    }
    r = c.post("/summarize", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "It runs."
    assert data["trace"]["translation"]["pivot_code"] == "def foo():\n    pass"


def test_summarize_validation_error_returns_422(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/summarize", json={"language": "ruby"})  # missing code+model
    assert r.status_code == 422
