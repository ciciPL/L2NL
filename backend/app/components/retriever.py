from __future__ import annotations
import json
import os
import re
from app.schemas import Example

# Ported from L2NL_release/03_retrieval/BM25.py (paper §3.3). BM25 over a cleaned
# CodeXGLUE / CSN Python corpus; query = the Python pivot code.


def makestr(lst: list[str]) -> str:
    if not lst:
        return ""
    return (" ".join(lst) + " ").replace(" .", ".").strip()


def clean_python_code(code: str, docstring: str = "") -> str:
    """Strip docstrings so retrieval can't leak the reference summary."""
    if not code:
        return ""
    code = re.sub(r'^\s*"""[\s\S]*?"""', '', code, flags=re.MULTILINE)
    code = re.sub(r"^\s*'''[\s\S]*?'''", '', code, flags=re.MULTILINE)
    if docstring:
        code = code.replace(docstring, "")
    return code.strip()


_CANNED = [
    Example(code="def add(a, b):\n    return a + b", core_blocks=[],
            summary="Returns the sum of two numbers.", score=1.0),
    Example(code="def is_even(n):\n    return n % 2 == 0", core_blocks=[],
            summary="Checks whether a number is even.", score=0.8),
]


class Retriever:
    """BM25 retrieval over a high-resource Python corpus (paper §3.3).

    With a corpus path, builds a resident BM25 index at construction. Without
    one (stub shell / local dev), returns canned examples so the pipeline runs.
    """

    def __init__(self, corpus_path: str | None = None, limit: int | None = None):
        self.corpus_path = corpus_path
        self.limit = limit
        self._bm25 = None
        self._codes: list[str] = []
        self._summaries: list[str] = []
        if corpus_path and os.path.exists(corpus_path):
            self._build_index(corpus_path)

    def _build_index(self, path: str) -> None:
        from rank_bm25 import BM25Okapi
        codes, summaries = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if self.limit and len(codes) >= self.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                js = json.loads(line)
                cleaned = clean_python_code(js.get("code", ""), js.get("docstring", ""))
                tokens = js.get("docstring_tokens", [])
                if not tokens and js.get("docstring"):
                    tokens = js["docstring"].split()
                codes.append(cleaned)
                summaries.append(makestr(tokens))
        self._codes = codes
        self._summaries = summaries
        self._bm25 = BM25Okapi([c.split() for c in codes])

    def retrieve(self, pivot_code: str, k: int) -> list[Example]:
        if self._bm25 is None:
            return (_CANNED * ((k // len(_CANNED)) + 1))[:k]
        scores = self._bm25.get_scores(clean_python_code(pivot_code).split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Example(code=self._codes[i], core_blocks=[],
                    summary=self._summaries[i], score=float(scores[i]))
            for i in top
        ]
