from __future__ import annotations
import hashlib


class Embedder:
    """SBERT wrapper. Stub uses a deterministic hash vector so the shell runs
    without downloading models. Replace _encode_one with sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", load: bool = False):
        self._model = None
        if load:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return self._model.encode(texts).tolist()
        return [self._encode_one(t) for t in texts]

    @staticmethod
    def _encode_one(text: str, dim: int = 16) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [h[i % len(h)] / 255.0 for i in range(dim)]
