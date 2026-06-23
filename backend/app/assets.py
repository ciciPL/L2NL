from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _entry_for(manifest: dict | None, kind: str) -> dict | None:
    if not manifest:
        return None
    for entry in manifest.get("assets", []):
        name = str(entry.get("name", ""))
        target = str(entry.get("target", ""))
        extract_to = str(entry.get("extractTo", ""))
        haystack = f"{name} {target} {extract_to}".lower()
        if kind == "extractor" and "extractor" in haystack and "pytorch_model.bin" in haystack:
            return entry
        if kind == "corpus" and "corpus" in haystack and "corpus_30k.jsonl" in haystack:
            return entry
        if kind == "codebert" and "codebert" in haystack:
            return entry
    return None


def _checksum(path: Path, entry: dict | None) -> bool | None:
    if not entry:
        return None
    expected_sha = entry.get("sha256")
    expected_size = entry.get("size")
    if not expected_sha or not isinstance(expected_size, int):
        return None
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return _sha256_file(path) == expected_sha


def _codebert_archive_path(settings, entry: dict | None) -> Path | None:
    if not settings.assets_manifest or not entry:
        return None
    target = entry.get("target") or entry.get("file") or entry.get("name")
    if not target:
        return None
    return Path(settings.assets_manifest).parent / str(target)


def _file_status(path: str | None, entry: dict | None = None) -> dict:
    p = Path(path) if path else None
    present = bool(p and p.is_file())
    checksum = _checksum(p, entry) if p else None
    return {"path": path or "", "present": present, "checksum": checksum}


def _dir_status(path: str | None, checksum_path: Path | None = None, entry: dict | None = None) -> dict:
    p = Path(path) if path else None
    present = bool(p and p.is_dir() and any(p.iterdir()))
    checksum = _checksum(checksum_path, entry) if checksum_path else None
    return {"path": path or "", "present": present, "checksum": checksum}


def _asset_ready(status: dict) -> bool:
    return bool(status["present"]) and status["checksum"] is not False


def asset_status(settings) -> dict:
    manifest = _load_manifest(settings.assets_manifest)
    codebert_entry = _entry_for(manifest, "codebert")
    assets = {
        "extractor": _file_status(settings.extractor_weights, _entry_for(manifest, "extractor")),
        "corpus": _file_status(settings.corpus_path, _entry_for(manifest, "corpus")),
        "codebert": _dir_status(
            settings.codebert_path,
            _codebert_archive_path(settings, codebert_entry),
            codebert_entry,
        ),
    }
    ready = all(_asset_ready(v) for v in assets.values())
    return {
        "ready": ready or settings.allow_stubs,
        "assets": assets,
        "allow_stubs": settings.allow_stubs,
        "manifest": settings.assets_manifest or "",
    }


def ensure_ready(settings) -> None:
    status = asset_status(settings)
    if status["ready"]:
        return
    missing = [k for k, v in status["assets"].items() if not v["present"]]
    mismatched = [k for k, v in status["assets"].items() if v["checksum"] is False]
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if mismatched:
        problems.append("checksum mismatch " + ", ".join(mismatched))
    raise RuntimeError(
        "production assets are not ready: "
        + "; ".join(problems)
        + ". Run setup, set CS_ALLOW_STUBS=1 for demo mode, or provide a local asset folder."
    )
