from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


ASSETS = [
    ("extractor/pytorch_model.bin", Path("extractor/pytorch_model.bin"), "extractor/pytorch_model.bin", None),
    ("corpus/corpus_30k.jsonl", Path("corpus/corpus_30k.jsonl"), "corpus/corpus_30k.jsonl", None),
    ("codebert-base", Path("codebert/codebert-base.tar.gz"), "codebert-base.tar.gz", "codebert-base"),
]


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_asset(src: Path, out_dir: Path, split_size: int) -> list[dict]:
    parts = []
    idx = 1
    with src.open("rb") as f:
        while True:
            data = f.read(split_size)
            if not data:
                break
            name = f"{src.name}.part{idx:03d}"
            dest = out_dir / name
            dest.write_bytes(data)
            parts.append({"file": name, "sha256": sha256_path(dest), "size": len(data)})
            idx += 1
    return parts


def build_release_assets(
    src_dir: Path,
    out_dir: Path,
    release: str,
    gitee_base_url: str,
    global_base_url: str | None = None,
    split_size: int = 60 * 1024 * 1024,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [{"id": "gitee", "baseUrl": gitee_base_url.rstrip("/"), "enabledByDefault": True}]
    if global_base_url:
        sources.append({"id": "global", "baseUrl": global_base_url.rstrip("/"), "global": True})

    manifest = {"version": 2, "release": release, "sources": sources, "assets": []}
    for name, rel, target, extract_to in ASSETS:
        src = src_dir / rel
        if not src.is_file():
            raise FileNotFoundError(f"missing required asset: {src}")
        entry = {
            "name": name,
            "target": target,
            "sha256": sha256_path(src),
            "size": src.stat().st_size,
            "parts": split_asset(src, out_dir, split_size),
        }
        if extract_to:
            entry["extractTo"] = extract_to
        manifest["assets"].append(entry)

    manifest_path = out_dir / "assets-manifest.v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_checksums(out_dir)
    return manifest


def write_checksums(out_dir: Path) -> None:
    lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{sha256_path(p)}  {p.name}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Build Gitee-friendly multipart Code Summary release assets.")
    p.add_argument("--src", required=True, type=Path, help="Directory containing extractor/, corpus/, and codebert/")
    p.add_argument("--out", required=True, type=Path, help="Output directory for release attachments")
    p.add_argument("--release", default="v0.2-assets-cn")
    p.add_argument("--gitee-base-url", required=True)
    p.add_argument("--global-base-url")
    p.add_argument("--split-size-mib", type=int, default=60)
    args = p.parse_args()
    build_release_assets(
        args.src, args.out, args.release, args.gitee_base_url, args.global_base_url,
        split_size=args.split_size_mib * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
