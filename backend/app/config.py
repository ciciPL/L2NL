from __future__ import annotations
import os


class Settings:
    corpus_path: str | None = os.getenv("CS_CORPUS_PATH")
    corpus_limit: int | None = (int(os.environ["CS_CORPUS_LIMIT"])
                                if os.getenv("CS_CORPUS_LIMIT") else None)
    extractor_weights: str | None = os.getenv("CS_EXTRACTOR_WEIGHTS")
    codebert_path: str = os.getenv("CS_CODEBERT_PATH", "microsoft/codebert-base")
    device: str | None = os.getenv("CS_DEVICE")  # e.g. "cuda" / "cpu"; None = auto
    load_sbert: bool = os.getenv("CS_LOAD_SBERT", "0") == "1"


settings = Settings()
