from __future__ import annotations
import os


class Settings:
    corpus_path: str | None = os.getenv("CS_CORPUS_PATH")
    extractor_weights: str | None = os.getenv("CS_EXTRACTOR_WEIGHTS")
    load_sbert: bool = os.getenv("CS_LOAD_SBERT", "0") == "1"


settings = Settings()
