from __future__ import annotations
import os
from pathlib import Path


_DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


class Settings:
    corpus_path: str | None = os.getenv("CS_CORPUS_PATH")
    corpus_limit: int | None = (int(os.environ["CS_CORPUS_LIMIT"])
                                if os.getenv("CS_CORPUS_LIMIT") else None)
    extractor_weights: str | None = os.getenv("CS_EXTRACTOR_WEIGHTS")
    codebert_path: str = os.getenv("CS_CODEBERT_PATH", str(_DEFAULT_ASSET_DIR / "codebert-base"))
    assets_manifest: str | None = os.getenv("CS_ASSETS_MANIFEST")
    device: str | None = os.getenv("CS_DEVICE")  # e.g. "cuda" / "cpu"; None = auto
    load_sbert: bool = os.getenv("CS_LOAD_SBERT", "0") == "1"
    allow_stubs: bool = os.getenv("CS_ALLOW_STUBS", "0") == "1"
    llm_timeout: float = float(os.getenv("CS_LLM_TIMEOUT", "60"))
    llm_max_tokens: int = int(os.getenv("CS_LLM_MAX_TOKENS", "512"))


settings = Settings()
