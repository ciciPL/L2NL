import json
from pathlib import Path

from deploy.build_assets_release import build_release_assets


def test_build_release_assets_writes_parts_manifest_and_checksums(tmp_path):
    src = tmp_path / "src"
    out = tmp_path / "out"
    (src / "extractor").mkdir(parents=True)
    (src / "corpus").mkdir()
    (src / "codebert").mkdir()
    (src / "extractor" / "pytorch_model.bin").write_text("abcdef")
    (src / "corpus" / "corpus_30k.jsonl").write_text("ghijkl")
    (src / "codebert" / "codebert-base.tar.gz").write_text("mnopqr")

    build_release_assets(
        src, out,
        release="v0.2-assets-cn",
        gitee_base_url="https://gitee.com/acme/l2nl/releases/download/v0.2-assets-cn",
        global_base_url="https://github.com/acme/l2nl/releases/download/v0.2-assets-cn",
        split_size=3,
    )

    manifest = json.loads((out / "assets-manifest.v2.json").read_text())
    assert manifest["version"] == 2
    assert manifest["sources"][0]["id"] == "gitee"
    assert manifest["sources"][0]["enabledByDefault"] is True
    assert manifest["sources"][1]["global"] is True
    first = manifest["assets"][0]
    assert first["name"] == "extractor/pytorch_model.bin"
    assert first["target"] == "extractor/pytorch_model.bin"
    assert [p["size"] for p in first["parts"]] == [3, 3]
    assert manifest["assets"][1]["target"] == "corpus/corpus_30k.jsonl"
    assert manifest["assets"][2]["target"] == "codebert-base.tar.gz"
    assert (out / "pytorch_model.bin.part001").read_text() == "abc"
    assert (out / "pytorch_model.bin.part002").read_text() == "def"
    sums = (out / "SHA256SUMS.txt").read_text()
    assert "assets-manifest.v2.json" in sums
    assert "pytorch_model.bin.part001" in sums
