import cli
from app.models.llm import FakeLLMClient


def test_run_file_prints_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "make_llm",
                        lambda base_url, api_key, model: FakeLLMClient(
                            responses=["<summary>CLI summary.</summary>"]))
    f = tmp_path / "snippet.rb"
    f.write_text("def foo; 1; end")
    cli.main(["--code-file", str(f), "--language", "ruby",
              "--base-url", "http://x/v1", "--model", "m"])
    out = capsys.readouterr().out
    assert "CLI summary." in out
