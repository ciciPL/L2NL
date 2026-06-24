from __future__ import annotations
import json
import os
import re
import tokenize
from io import StringIO
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


def _split_identifier(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in re.split(r"[^A-Za-z0-9]+", text):
        if not chunk:
            continue
        chunk = re.sub(r"([A-Za-z]+)(\d+)$", r"\1 \2", chunk)
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", chunk)
        split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split)
        parts.extend(p.lower() for p in split.split() if p)
    return parts


_MEANINGFUL_OPS = {
    "+", "-", "*", "/", "//", "%", "**",
    "==", "!=", "<=", ">=", "<", ">",
}

_CODE_STOPWORDS = {
    "and", "as", "assert", "async", "await", "break", "case", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for",
    "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    "self", "cls", "true", "false", "none",
}


def _is_index_token(token: str) -> bool:
    if not token:
        return False
    if token in _CODE_STOPWORDS:
        return False
    if token in _MEANINGFUL_OPS:
        return True
    if token.isdigit():
        return len(token) > 1
    if len(token) == 1 and token.isalpha():
        return False
    return any(ch.isalnum() for ch in token)


def tokenize_code_for_bm25(code: str) -> list[str]:
    """Tokenize code for BM25 using Python syntax and identifier subwords."""
    cleaned = clean_python_code(code)
    if not cleaned:
        return []
    out: list[str] = []
    try:
        stream = StringIO(cleaned).readline
        for tok in tokenize.generate_tokens(stream):
            if tok.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            }:
                continue
            text = tok.string.strip()
            if not text:
                continue
            if tok.type == tokenize.NAME:
                out.extend(_split_identifier(text))
            elif tok.type == tokenize.OP:
                if text in _MEANINGFUL_OPS:
                    out.append(text)
            else:
                out.extend(_split_identifier(text))
    except (tokenize.TokenError, IndentationError):
        for text in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s\w]", cleaned):
            if re.match(r"[A-Za-z_]", text):
                out.extend(_split_identifier(text))
            elif text in _MEANINGFUL_OPS:
                out.append(text.lower())
    return [t for t in out if _is_index_token(t)]


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

    def __init__(self, corpus_path: str | None = None, limit: int | None = None,
                 allow_stubs: bool = False):
        self.corpus_path = corpus_path
        self.limit = limit
        self.allow_stubs = allow_stubs
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
        self._bm25 = BM25Okapi([tokenize_code_for_bm25(c) for c in codes])

    def retrieve(self, pivot_code: str, k: int) -> list[Example]:
        if self._bm25 is None:
            if not self.allow_stubs:
                raise RuntimeError(
                    "CS_CORPUS_PATH is missing or invalid; run setup or set CS_ALLOW_STUBS=1 for demo mode"
                )
            return (_CANNED * ((k // len(_CANNED)) + 1))[:k]
        scores = self._bm25.get_scores(tokenize_code_for_bm25(pivot_code))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            Example(code=self._codes[i], core_blocks=[],
                    summary=self._summaries[i], score=float(scores[i]))
            for i in top
        ]
