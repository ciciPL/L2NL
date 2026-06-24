import cli
import pytest
from app.models.llm import FakeLLMClient
from app.schemas import SummarizeResponse


def test_run_file_prints_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "make_llm",
                        lambda base_url, api_key, model: FakeLLMClient(
                            responses=["<summary>CLI summary.</summary>"]))
    monkeypatch.setattr(cli.settings, "allow_stubs", True)
    f = tmp_path / "snippet.rb"
    f.write_text("def foo; 1; end")
    cli.main(["--code-file", str(f), "--language", "ruby",
              "--base-url", "http://x/v1", "--model", "m"])
    out = capsys.readouterr().out
    assert "CLI summary." in out


def test_cli_requires_production_assets_before_pipeline(tmp_path, monkeypatch):
    def fail_make_llm(*_args):
        raise AssertionError("LLM should not be constructed before asset readiness")

    monkeypatch.setattr(cli, "make_llm", fail_make_llm)
    monkeypatch.setattr(cli.settings, "allow_stubs", False)
    monkeypatch.setattr(cli.settings, "extractor_weights", str(tmp_path / "missing.bin"))
    monkeypatch.setattr(cli.settings, "corpus_path", str(tmp_path / "missing.jsonl"))
    monkeypatch.setattr(cli.settings, "codebert_path", str(tmp_path / "missing-codebert"))
    monkeypatch.setattr(cli.settings, "assets_manifest", None)
    f = tmp_path / "snippet.rb"
    f.write_text("def foo; 1; end")

    with pytest.raises(RuntimeError, match="production assets are not ready"):
        cli.main(["--code-file", str(f), "--language", "ruby",
                  "--base-url", "http://x/v1", "--model", "m"])


def test_cli_defaults_to_paper_top_three_retrieval(tmp_path, monkeypatch):
    seen: dict[str, int] = {}

    class FakePipeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _code, _language, params, trace):
            seen["k"] = params.k
            return SummarizeResponse(summary="CLI summary.")

    monkeypatch.setattr(cli, "ensure_ready", lambda _settings: None)
    monkeypatch.setattr(cli, "make_llm", lambda *_args: object())
    monkeypatch.setattr(cli, "Embedder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "Translator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "Retriever", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "Extractor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "Generator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    f = tmp_path / "snippet.rb"
    f.write_text("def foo; 1; end")
    cli.main(["--code-file", str(f), "--language", "ruby",
              "--base-url", "http://x/v1", "--model", "m"])

    assert seen["k"] == 3
