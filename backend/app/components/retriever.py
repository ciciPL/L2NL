from __future__ import annotations
from app.schemas import Example


class Retriever:
    """BM25 retrieval over a high-resource parallel corpus.

    STUB: returns canned examples. Replace `__init__` to build a BM25 index from
    corpus_path and `retrieve` to query it. Signatures are final.
    """

    def __init__(self, corpus_path: str | None = None):
        self.corpus_path = corpus_path

    def retrieve(self, pivot_code: str, k: int) -> list[Example]:
        canned = [
            Example(
                code="def add(a, b):\n    return a + b",
                core_blocks=[],
                summary="Returns the sum of two numbers.",
                score=1.0,
            ),
            Example(
                code="def is_even(n):\n    return n % 2 == 0",
                core_blocks=[],
                summary="Checks whether a number is even.",
                score=0.8,
            ),
        ]
        return (canned * ((k // len(canned)) + 1))[:k]
