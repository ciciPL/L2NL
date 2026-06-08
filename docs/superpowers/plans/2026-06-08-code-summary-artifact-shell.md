# Low-Resource Code Summary Artifact — Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable end-to-end "shell": a FastAPI backend that implements the paper's 4-stage pipeline orchestration with stubbed components + a unified OpenAI-compatible LLM client, plus a VS Code extension that sends selected code to the backend and renders the summary + per-stage trace. Real pipeline implementations plug into the component files later.

**Architecture:** VS Code thin client (TypeScript) ↔ local FastAPI backend (Python) over HTTP. Backend hosts the pipeline as 4 injectable components (translator / retriever / extractor / generator). Online vs offline LLM is a single OpenAI-compatible client differing only by `base_url`. Components ship as stubs returning plausible data so the whole thing runs immediately; the generator and LLM client are real so a real model produces a real summary.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, pydantic v2, openai SDK, rank-bm25, sentence-transformers, tree-sitter (deps declared but only used by real impls later), pytest. TypeScript, VS Code Extension API, esbuild, vitest.

---

## File Structure

```
backend/
  pyproject.toml
  app/
    __init__.py
    config.py          # Settings (env-driven), asset paths
    schemas.py         # pydantic request/response/trace models
    models/
      __init__.py
      llm.py           # LLMClient (OpenAI-compatible) + FakeLLMClient
      embedder.py      # Embedder (SBERT) stub
    components/
      __init__.py
      translator.py    # Translator stub (echo pivot)
      retriever.py     # Retriever stub (canned examples)
      extractor.py     # Extractor stub (naive blocks)
      generator.py     # Generator REAL (prompt assembly + tag extraction)
    pipeline.py        # Pipeline orchestration -> Trace
    main.py            # FastAPI app: GET /health, POST /summarize
  cli.py               # Standalone reproducible entrypoint
  tests/
    test_schemas.py
    test_llm.py
    test_generator.py
    test_pipeline.py
    test_api.py
    test_cli.py
extension/
  package.json
  tsconfig.json
  esbuild.js
  src/
    config.ts          # pure buildRequest(settings) + activeModel()
    client.ts          # postSummarize(backendUrl, body)
    panel.ts           # webview HTML for summary + trace
    extension.ts       # activate: command, status bar, backend lifecycle
  test/
    config.test.ts
```

---

## Task 1: Backend scaffold

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/app/__init__.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/components/__init__.py`
- Create: `backend/tests/__init__.py`

- [ ] **Step 1: Create `backend/pyproject.toml`**

```toml
[project]
name = "code-summary-backend"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "pydantic>=2.6",
  "openai>=1.30",
  "rank-bm25>=0.2.2",
  "sentence-transformers>=2.7",
  "tree-sitter>=0.21",
]

[project.optional-dependencies]
dev = ["pytest>=8", "httpx>=0.27"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package markers**

Create `backend/app/__init__.py`, `backend/app/models/__init__.py`, `backend/app/components/__init__.py`, `backend/tests/__init__.py` each as empty files.

- [ ] **Step 3: Install and verify**

Run: `cd backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`
Expected: installs without error.

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/app backend/tests
git commit -m "chore: backend scaffold and dependencies"
```

---

## Task 2: Schemas

**Files:**
- Create: `backend/app/schemas.py`
- Test: `backend/tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_schemas.py
from app.schemas import (
    CoreBlock, Example, TranslationResult, Trace,
    SummarizeRequest, SummarizeResponse, ModelConfig, Params,
)


def test_request_roundtrip_with_defaults():
    req = SummarizeRequest(
        code="def foo; end",
        language="ruby",
        model=ModelConfig(base_url="http://x/v1", api_key="k", model="m"),
    )
    assert req.params.k == 5
    assert req.params.temperatures == [0.0, 0.4, 0.8]
    assert req.trace is True


def test_response_serialization():
    tr = TranslationResult(
        pivot_code="def foo(): pass",
        candidates=["def foo(): pass"],
        selected_score=1.0, repaired=False, fell_back=False,
    )
    trace = Trace(translation=tr, retrieved=[], core_blocks=[], prompt="P")
    resp = SummarizeResponse(summary="S", trace=trace)
    dumped = resp.model_dump()
    assert dumped["summary"] == "S"
    assert dumped["trace"]["translation"]["pivot_code"] == "def foo(): pass"
    assert dumped["error"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Write `backend/app/schemas.py`**

```python
from __future__ import annotations
from pydantic import BaseModel, Field


class CoreBlock(BaseModel):
    text: str
    block_type: str
    prob: float


class Example(BaseModel):
    code: str
    core_blocks: list[CoreBlock] = Field(default_factory=list)
    summary: str
    score: float


class TranslationResult(BaseModel):
    pivot_code: str
    candidates: list[str] = Field(default_factory=list)
    selected_score: float
    repaired: bool
    fell_back: bool


class Trace(BaseModel):
    translation: TranslationResult
    retrieved: list[Example] = Field(default_factory=list)
    core_blocks: list[CoreBlock] = Field(default_factory=list)
    prompt: str


class ModelConfig(BaseModel):
    base_url: str
    api_key: str = "sk-no-key"
    model: str


class Params(BaseModel):
    k: int = 5
    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.4, 0.8])
    lambda_: float = Field(default=0.5, alias="lambda")
    threshold: float = 0.5
    max_repair_iters: int = 3

    model_config = {"populate_by_name": True}


class SummarizeRequest(BaseModel):
    code: str
    language: str
    model: ModelConfig
    params: Params = Field(default_factory=Params)
    trace: bool = True


class SummarizeResponse(BaseModel):
    summary: str
    trace: Trace | None = None
    error: str | None = None
    failed_stage: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_schemas.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_schemas.py
git commit -m "feat: pydantic schemas for summarize request/response/trace"
```

---

## Task 3: LLM client + fake

**Files:**
- Create: `backend/app/models/llm.py`
- Test: `backend/tests/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_llm.py
from app.models.llm import FakeLLMClient


def test_fake_client_returns_tagged_summary_by_default():
    c = FakeLLMClient()
    out = c.chat([{"role": "user", "content": "anything"}], temperature=0.0)
    assert "<summary>" in out and "</summary>" in out


def test_fake_client_records_calls():
    c = FakeLLMClient(responses=["A", "B"])
    assert c.chat([{"role": "user", "content": "x"}], temperature=0.2) == "A"
    assert c.chat([{"role": "user", "content": "y"}], temperature=0.7) == "B"
    assert len(c.calls) == 2
    assert c.calls[0]["temperature"] == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.llm'`.

- [ ] **Step 3: Write `backend/app/models/llm.py`**

```python
from __future__ import annotations


class LLMClient:
    """OpenAI-compatible chat client. Online vs offline differ only by base_url."""

    def __init__(self, base_url: str, api_key: str, model: str):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key or "sk-no-key")

    def chat(self, messages: list[dict], temperature: float) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class FakeLLMClient(LLMClient):
    """Deterministic client for tests/stubs. No network."""

    def __init__(self, responses: list[str] | None = None):
        self.model = "fake"
        self._responses = list(responses) if responses else None
        self._i = 0
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], temperature: float) -> str:
        self.calls.append({"messages": messages, "temperature": temperature})
        if self._responses is not None:
            out = self._responses[min(self._i, len(self._responses) - 1)]
            self._i += 1
            return out
        return "<summary>Stubbed summary of the provided code.</summary>"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_llm.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/models/llm.py backend/tests/test_llm.py
git commit -m "feat: OpenAI-compatible LLM client + deterministic fake"
```

---

## Task 4: Embedder stub

**Files:**
- Create: `backend/app/models/embedder.py`

- [ ] **Step 1: Write `backend/app/models/embedder.py`**

```python
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
```

- [ ] **Step 2: Smoke check**

Run: `cd backend && python -c "from app.models.embedder import Embedder; print(len(Embedder().encode(['a','b'])))"`
Expected: prints `2`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/models/embedder.py
git commit -m "feat: embedder stub (deterministic hash vectors)"
```

---

## Task 5: Translator stub

**Files:**
- Create: `backend/app/components/translator.py`

- [ ] **Step 1: Write `backend/app/components/translator.py`**

```python
from __future__ import annotations
from app.schemas import TranslationResult, Params
from app.models.llm import LLMClient
from app.models.embedder import Embedder


class Translator:
    """Cross-language translation to a high-resource pivot language.

    STUB: echoes the input as pivot code. Replace `translate` with the paper's
    multi-temperature sampling + AST validate/repair + back-translation selection.
    Constructor signature is final; fill the body with existing code.
    """

    def __init__(self, llm: LLMClient, embedder: Embedder, params: Params):
        self.llm = llm
        self.embedder = embedder
        self.params = params

    def translate(self, code: str, src_lang: str) -> TranslationResult:
        return TranslationResult(
            pivot_code=code,
            candidates=[code],
            selected_score=1.0,
            repaired=False,
            fell_back=False,
        )
```

- [ ] **Step 2: Smoke check**

Run: `cd backend && python -c "from app.components.translator import Translator; from app.models.llm import FakeLLMClient; from app.models.embedder import Embedder; from app.schemas import Params; print(Translator(FakeLLMClient(), Embedder(), Params()).translate('x','ruby').pivot_code)"`
Expected: prints `x`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/components/translator.py
git commit -m "feat: translator component stub with final interface"
```

---

## Task 6: Retriever stub

**Files:**
- Create: `backend/app/components/retriever.py`

- [ ] **Step 1: Write `backend/app/components/retriever.py`**

```python
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
```

- [ ] **Step 2: Smoke check**

Run: `cd backend && python -c "from app.components.retriever import Retriever; print(len(Retriever().retrieve('q', 5)))"`
Expected: prints `5`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/components/retriever.py
git commit -m "feat: retriever component stub with final interface"
```

---

## Task 7: Extractor stub

**Files:**
- Create: `backend/app/components/extractor.py`

- [ ] **Step 1: Write `backend/app/components/extractor.py`**

```python
from __future__ import annotations
from app.schemas import CoreBlock


class Extractor:
    """Core-statement-block extraction: AST split + trained binary classifier.

    STUB: splits by non-empty lines and labels each as 'other' with prob 1.0.
    Replace `__init__` to load classifier weights and `extract` with AST semantic
    splitting + classifier scoring + threshold filtering. Signatures are final.
    """

    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path

    def extract(self, code: str, threshold: float) -> list[CoreBlock]:
        blocks: list[CoreBlock] = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped:
                blocks.append(
                    CoreBlock(text=stripped, block_type="other", prob=1.0)
                )
        return blocks
```

- [ ] **Step 2: Smoke check**

Run: `cd backend && python -c "from app.components.extractor import Extractor; print(len(Extractor().extract('a\n\nb', 0.5)))"`
Expected: prints `2`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/components/extractor.py
git commit -m "feat: extractor component stub with final interface"
```

---

## Task 8: Generator (real prompt assembly)

**Files:**
- Create: `backend/app/components/generator.py`
- Test: `backend/tests/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_generator.py
from app.components.generator import Generator
from app.models.llm import FakeLLMClient
from app.schemas import Example, CoreBlock


def test_generate_extracts_tagged_summary_and_returns_prompt():
    llm = FakeLLMClient(responses=["noise <summary>It adds two numbers.</summary> trailing"])
    gen = Generator(llm)
    examples = [Example(code="def add(a,b): return a+b",
                        core_blocks=[CoreBlock(text="return a+b", block_type="other", prob=1.0)],
                        summary="Adds two numbers.", score=1.0)]
    blocks = [CoreBlock(text="return a+b", block_type="other", prob=1.0)]
    summary, prompt = gen.generate("def add(a,b): return a+b", blocks, examples)
    assert summary == "It adds two numbers."
    assert "<code>" in prompt and "<summary>" in prompt
    assert "Adds two numbers." in prompt  # few-shot reference present


def test_generate_falls_back_to_full_text_when_no_tags():
    llm = FakeLLMClient(responses=["plain summary no tags"])
    summary, _ = Generator(llm).generate("code", [], [])
    assert summary == "plain summary no tags"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.components.generator'`.

- [ ] **Step 3: Write `backend/app/components/generator.py`**

```python
from __future__ import annotations
import re
from app.schemas import Example, CoreBlock
from app.models.llm import LLMClient

_SYSTEM = (
    "You are an expert programmer. Generate a concise one-sentence natural-"
    "language summary of the given code. The provided core statement blocks "
    "highlight the key behavior. Output ONLY the summary wrapped in "
    "<summary></summary> tags."
)
_INSTRUCTION = "Summarize what the code does in one sentence."
_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def _blocks_text(blocks: list[CoreBlock]) -> str:
    return "\n".join(b.text for b in blocks)


class Generator:
    """Structure-guided summary generation with tagged boundaries (real impl)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _build_prompt(self, pivot_code: str, core_blocks: list[CoreBlock],
                      examples: list[Example]) -> str:
        parts: list[str] = []
        for ex in examples:
            parts.append(
                f"<code>\n{ex.code}\n</code>\n"
                f"<core>\n{_blocks_text(ex.core_blocks)}\n</core>\n"
                f"<instruction>{_INSTRUCTION}</instruction>\n"
                f"<summary>{ex.summary}</summary>"
            )
        parts.append(
            f"<code>\n{pivot_code}\n</code>\n"
            f"<core>\n{_blocks_text(core_blocks)}\n</core>\n"
            f"<instruction>{_INSTRUCTION}</instruction>\n"
            f"<summary>"
        )
        return "\n\n".join(parts)

    def generate(self, pivot_code: str, core_blocks: list[CoreBlock],
                 examples: list[Example]) -> tuple[str, str]:
        prompt = self._build_prompt(pivot_code, core_blocks, examples)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, temperature=0.0)
        m = _TAG_RE.search(raw)
        summary = m.group(1).strip() if m else raw.strip()
        return summary, prompt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_generator.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/components/generator.py backend/tests/test_generator.py
git commit -m "feat: structure-guided generator with tagged prompt + extraction"
```

---

## Task 9: Pipeline orchestration

**Files:**
- Create: `backend/app/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_pipeline.py
from app.pipeline import Pipeline
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.models.llm import FakeLLMClient
from app.models.embedder import Embedder
from app.schemas import Params


def _pipeline(llm):
    emb = Embedder()
    params = Params()
    return Pipeline(
        translator=Translator(llm, emb, params),
        retriever=Retriever(),
        extractor=Extractor(),
        generator=Generator(llm),
    )


def test_run_returns_summary_and_full_trace():
    llm = FakeLLMClient(responses=["<summary>It does X.</summary>"])
    resp = _pipeline(llm).run("def foo(): return 1", "ruby", Params(), trace=True)
    assert resp.error is None
    assert resp.summary == "It does X."
    assert resp.trace is not None
    assert resp.trace.translation.pivot_code == "def foo(): return 1"
    assert len(resp.trace.retrieved) == Params().k
    assert resp.trace.prompt.startswith("<code>") or "<code>" in resp.trace.prompt


def test_run_without_trace_omits_trace():
    llm = FakeLLMClient(responses=["<summary>Y.</summary>"])
    resp = _pipeline(llm).run("x", "ruby", Params(), trace=False)
    assert resp.summary == "Y."
    assert resp.trace is None


def test_stage_failure_is_reported():
    class BoomRetriever(Retriever):
        def retrieve(self, pivot_code, k):
            raise RuntimeError("index missing")

    llm = FakeLLMClient(responses=["<summary>Z.</summary>"])
    emb = Embedder(); params = Params()
    pipe = Pipeline(
        translator=Translator(llm, emb, params),
        retriever=BoomRetriever(),
        extractor=Extractor(),
        generator=Generator(llm),
    )
    resp = pipe.run("x", "ruby", params, trace=True)
    assert resp.error is not None
    assert resp.failed_stage == "retrieve"
    assert "index missing" in resp.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline'`.

- [ ] **Step 3: Write `backend/app/pipeline.py`**

```python
from __future__ import annotations
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.schemas import Params, Trace, SummarizeResponse


class Pipeline:
    def __init__(self, translator: Translator, retriever: Retriever,
                 extractor: Extractor, generator: Generator):
        self.translator = translator
        self.retriever = retriever
        self.extractor = extractor
        self.generator = generator

    def run(self, code: str, language: str, params: Params,
            trace: bool) -> SummarizeResponse:
        stage = "translate"
        try:
            translation = self.translator.translate(code, language)

            stage = "retrieve"
            examples = self.retriever.retrieve(translation.pivot_code, params.k)

            stage = "extract"
            core_blocks = self.extractor.extract(
                translation.pivot_code, params.threshold)
            for ex in examples:
                ex.core_blocks = self.extractor.extract(ex.code, params.threshold)

            stage = "generate"
            summary, prompt = self.generator.generate(
                translation.pivot_code, core_blocks, examples)
        except Exception as e:  # noqa: BLE001 - boundary: report which stage broke
            return SummarizeResponse(
                summary="", trace=None, error=str(e), failed_stage=stage)

        trace_obj = None
        if trace:
            trace_obj = Trace(
                translation=translation, retrieved=examples,
                core_blocks=core_blocks, prompt=prompt)
        return SummarizeResponse(summary=summary, trace=trace_obj)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: pipeline orchestration with per-stage error reporting"
```

---

## Task 10: Config + FastAPI app

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_api.py
from fastapi.testclient import TestClient
import app.main as main
from app.models.llm import FakeLLMClient


def _client(monkeypatch):
    # Force the app to use a deterministic LLM regardless of request model config.
    monkeypatch.setattr(main, "make_llm",
                        lambda cfg: FakeLLMClient(responses=["<summary>It runs.</summary>"]))
    return TestClient(main.app)


def test_health(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_summarize_returns_summary_and_trace(monkeypatch):
    c = _client(monkeypatch)
    body = {
        "code": "def foo; 1; end",
        "language": "ruby",
        "model": {"base_url": "http://x/v1", "api_key": "k", "model": "m"},
        "trace": True,
    }
    r = c.post("/summarize", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "It runs."
    assert data["trace"]["translation"]["pivot_code"] == "def foo; 1; end"


def test_summarize_validation_error_returns_422(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/summarize", json={"language": "ruby"})  # missing code+model
    assert r.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 3: Write `backend/app/config.py`**

```python
from __future__ import annotations
import os


class Settings:
    corpus_path: str | None = os.getenv("CS_CORPUS_PATH")
    extractor_weights: str | None = os.getenv("CS_EXTRACTOR_WEIGHTS")
    load_sbert: bool = os.getenv("CS_LOAD_SBERT", "0") == "1"


settings = Settings()
```

- [ ] **Step 4: Write `backend/app/main.py`**

```python
from __future__ import annotations
from fastapi import FastAPI
from app.config import settings
from app.schemas import SummarizeRequest, SummarizeResponse, ModelConfig
from app.models.llm import LLMClient
from app.models.embedder import Embedder
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.pipeline import Pipeline

app = FastAPI(title="Code Summary Backend")

# Resident resources loaded once at import (process start).
embedder = Embedder(load=settings.load_sbert)
retriever = Retriever(settings.corpus_path)
extractor = Extractor(settings.extractor_weights)


def make_llm(cfg: ModelConfig) -> LLMClient:
    return LLMClient(base_url=cfg.base_url, api_key=cfg.api_key, model=cfg.model)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": embedder is not None}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest) -> SummarizeResponse:
    llm = make_llm(req.model)
    pipeline = Pipeline(
        translator=Translator(llm, embedder, req.params),
        retriever=retriever,
        extractor=extractor,
        generator=Generator(llm),
    )
    return pipeline.run(req.code, req.language, req.params, req.trace)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && pytest tests/test_api.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/main.py backend/tests/test_api.py
git commit -m "feat: FastAPI app with /health and /summarize"
```

---

## Task 11: CLI entrypoint

**Files:**
- Create: `backend/cli.py`
- Test: `backend/tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_cli.py
import cli
from app.models.llm import FakeLLMClient


def test_run_file_prints_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "make_llm",
                        lambda base_url, api_key, model: FakeLLMClient(
                            responses=["<summary>CLI summary.</summary>"]))
    f = tmp_path / "snippet.rb"
    f.write_text("def foo; 1; end")
    cli.main(["--code-file", str(f), "--language", "ruby",
              "--base-url", "http://x/v1", "--model", "m"])
    out = capsys.readouterr().out
    assert "CLI summary." in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cli'`.

- [ ] **Step 3: Write `backend/cli.py`**

```python
from __future__ import annotations
import argparse
from pathlib import Path
from app.config import settings
from app.models.llm import LLMClient
from app.models.embedder import Embedder
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.pipeline import Pipeline
from app.schemas import Params


def make_llm(base_url: str, api_key: str, model: str) -> LLMClient:
    return LLMClient(base_url=base_url, api_key=api_key, model=model)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Low-resource code summary (paper artifact).")
    p.add_argument("--code-file", required=True)
    p.add_argument("--language", default="ruby")
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="sk-no-key")
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--no-trace", action="store_true")
    args = p.parse_args(argv)

    code = Path(args.code_file).read_text()
    llm = make_llm(args.base_url, args.api_key, args.model)
    params = Params(k=args.k)
    pipeline = Pipeline(
        translator=Translator(llm, Embedder(load=settings.load_sbert), params),
        retriever=Retriever(settings.corpus_path),
        extractor=Extractor(settings.extractor_weights),
        generator=Generator(llm),
    )
    resp = pipeline.run(code, args.language, params, trace=not args.no_trace)
    if resp.error:
        print(f"[FAILED at {resp.failed_stage}] {resp.error}")
        return
    print("=== SUMMARY ===")
    print(resp.summary)
    if resp.trace:
        print("\n=== PIVOT ===")
        print(resp.trace.translation.pivot_code)
        print(f"\n=== RETRIEVED ({len(resp.trace.retrieved)}) ===")
        for ex in resp.trace.retrieved:
            print(f"  [{ex.score:.2f}] {ex.summary}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_cli.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/cli.py backend/tests/test_cli.py
git commit -m "feat: standalone CLI entrypoint for reproducible runs"
```

---

## Task 12: Extension scaffold

**Files:**
- Create: `extension/package.json`
- Create: `extension/tsconfig.json`
- Create: `extension/esbuild.js`

- [ ] **Step 1: Write `extension/package.json`**

```json
{
  "name": "code-summary",
  "displayName": "Low-Resource Code Summary",
  "description": "Pivot-retrieval code summarization (paper artifact).",
  "version": "0.1.0",
  "engines": { "vscode": "^1.88.0" },
  "categories": ["Other"],
  "activationEvents": [],
  "main": "./dist/extension.js",
  "contributes": {
    "commands": [
      { "command": "codeSummary.summarizeSelection", "title": "Code Summary: Summarize Selection" },
      { "command": "codeSummary.startBackend", "title": "Code Summary: Start Backend" },
      { "command": "codeSummary.stopBackend", "title": "Code Summary: Stop Backend" },
      { "command": "codeSummary.toggleMode", "title": "Code Summary: Toggle Online/Offline" }
    ],
    "menus": {
      "editor/context": [
        { "command": "codeSummary.summarizeSelection", "when": "editorHasSelection", "group": "navigation" }
      ]
    },
    "configuration": {
      "title": "Code Summary",
      "properties": {
        "codeSummary.mode": { "type": "string", "enum": ["online", "offline"], "default": "online" },
        "codeSummary.online.baseUrl": { "type": "string", "default": "https://api.openai.com/v1" },
        "codeSummary.online.apiKey": { "type": "string", "default": "" },
        "codeSummary.online.model": { "type": "string", "default": "gpt-4o-mini" },
        "codeSummary.offline.baseUrl": { "type": "string", "default": "http://localhost:8080/v1" },
        "codeSummary.offline.model": { "type": "string", "default": "local-model" },
        "codeSummary.params.k": { "type": "number", "default": 5 },
        "codeSummary.params.temperatures": { "type": "array", "default": [0, 0.4, 0.8] },
        "codeSummary.params.lambda": { "type": "number", "default": 0.5 },
        "codeSummary.params.threshold": { "type": "number", "default": 0.5 },
        "codeSummary.params.maxRepairIters": { "type": "number", "default": 3 },
        "codeSummary.backend.url": { "type": "string", "default": "http://localhost:8000" },
        "codeSummary.backend.autoStart": { "type": "boolean", "default": true },
        "codeSummary.backend.pythonPath": { "type": "string", "default": "python" },
        "codeSummary.backend.cwd": { "type": "string", "default": "" }
      }
    }
  },
  "scripts": {
    "build": "node esbuild.js",
    "test": "vitest run"
  },
  "devDependencies": {
    "@types/vscode": "^1.88.0",
    "@types/node": "^20",
    "esbuild": "^0.21",
    "typescript": "^5.4",
    "vitest": "^1.6"
  }
}
```

- [ ] **Step 2: Write `extension/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2021",
    "module": "commonjs",
    "lib": ["ES2021"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "dist",
    "rootDir": "src"
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Write `extension/esbuild.js`**

```js
const esbuild = require("esbuild");
esbuild.build({
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  target: "node18",
}).catch(() => process.exit(1));
```

- [ ] **Step 4: Install deps**

Run: `cd extension && npm install`
Expected: installs without error.

- [ ] **Step 5: Commit**

```bash
git add extension/package.json extension/tsconfig.json extension/esbuild.js extension/package-lock.json
git commit -m "chore: VS Code extension scaffold and manifest"
```

---

## Task 13: Config mapping (pure, testable)

**Files:**
- Create: `extension/src/config.ts`
- Test: `extension/test/config.test.ts`
- Create: `extension/vitest.config.ts`

- [ ] **Step 1: Write `extension/vitest.config.ts`**

```ts
import { defineConfig } from "vitest/config";
export default defineConfig({ test: { include: ["test/**/*.test.ts"] } });
```

- [ ] **Step 2: Write the failing test**

```ts
// extension/test/config.test.ts
import { describe, it, expect } from "vitest";
import { buildRequest, RawSettings } from "../src/config";

const base: RawSettings = {
  mode: "online",
  online: { baseUrl: "https://api.openai.com/v1", apiKey: "K", model: "gpt-4o-mini" },
  offline: { baseUrl: "http://localhost:8080/v1", model: "local-model" },
  params: { k: 5, temperatures: [0, 0.4, 0.8], lambda: 0.5, threshold: 0.5, maxRepairIters: 3 },
};

describe("buildRequest", () => {
  it("uses the online model block when mode=online", () => {
    const req = buildRequest("def foo; end", "ruby", base);
    expect(req.model.base_url).toBe("https://api.openai.com/v1");
    expect(req.model.api_key).toBe("K");
    expect(req.model.model).toBe("gpt-4o-mini");
    expect(req.params.k).toBe(5);
    expect(req.trace).toBe(true);
  });

  it("uses the offline model block when mode=offline", () => {
    const req = buildRequest("x", "ruby", { ...base, mode: "offline" });
    expect(req.model.base_url).toBe("http://localhost:8080/v1");
    expect(req.model.api_key).toBe("sk-no-key");
    expect(req.model.model).toBe("local-model");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd extension && npx vitest run`
Expected: FAIL — cannot find module `../src/config`.

- [ ] **Step 4: Write `extension/src/config.ts`**

```ts
export interface RawSettings {
  mode: "online" | "offline";
  online: { baseUrl: string; apiKey: string; model: string };
  offline: { baseUrl: string; model: string };
  params: {
    k: number; temperatures: number[]; lambda: number;
    threshold: number; maxRepairIters: number;
  };
}

export interface SummarizeRequest {
  code: string;
  language: string;
  model: { base_url: string; api_key: string; model: string };
  params: {
    k: number; temperatures: number[]; lambda: number;
    threshold: number; max_repair_iters: number;
  };
  trace: boolean;
}

export function activeModel(s: RawSettings) {
  return s.mode === "offline"
    ? { base_url: s.offline.baseUrl, api_key: "sk-no-key", model: s.offline.model }
    : { base_url: s.online.baseUrl, api_key: s.online.apiKey, model: s.online.model };
}

export function buildRequest(
  code: string, language: string, s: RawSettings,
): SummarizeRequest {
  return {
    code,
    language,
    model: activeModel(s),
    params: {
      k: s.params.k,
      temperatures: s.params.temperatures,
      lambda: s.params.lambda,
      threshold: s.params.threshold,
      max_repair_iters: s.params.maxRepairIters,
    },
    trace: true,
  };
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd extension && npx vitest run`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add extension/src/config.ts extension/test/config.test.ts extension/vitest.config.ts
git commit -m "feat: pure config->request mapping with tests"
```

---

## Task 14: Backend HTTP client

**Files:**
- Create: `extension/src/client.ts`

- [ ] **Step 1: Write `extension/src/client.ts`**

```ts
import { SummarizeRequest } from "./config";

export interface CoreBlock { text: string; block_type: string; prob: number; }
export interface Example {
  code: string; core_blocks: CoreBlock[]; summary: string; score: number;
}
export interface Trace {
  translation: {
    pivot_code: string; candidates: string[]; selected_score: number;
    repaired: boolean; fell_back: boolean;
  };
  retrieved: Example[];
  core_blocks: CoreBlock[];
  prompt: string;
}
export interface SummarizeResponse {
  summary: string; trace: Trace | null;
  error: string | null; failed_stage: string | null;
}

export async function getHealth(backendUrl: string): Promise<boolean> {
  try {
    const r = await fetch(`${backendUrl}/health`);
    return r.ok;
  } catch {
    return false;
  }
}

export async function postSummarize(
  backendUrl: string, body: SummarizeRequest,
): Promise<SummarizeResponse> {
  const r = await fetch(`${backendUrl}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    throw new Error(`Backend returned ${r.status}: ${await r.text()}`);
  }
  return (await r.json()) as SummarizeResponse;
}
```

- [ ] **Step 2: Type-check**

Run: `cd extension && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add extension/src/client.ts
git commit -m "feat: backend HTTP client (health + summarize)"
```

---

## Task 15: Webview panel rendering

**Files:**
- Create: `extension/src/panel.ts`

- [ ] **Step 1: Write `extension/src/panel.ts`**

```ts
import * as vscode from "vscode";
import { SummarizeResponse } from "./client";

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderBody(resp: SummarizeResponse): string {
  if (resp.error) {
    return `<h2>Failed at stage: ${esc(resp.failed_stage ?? "unknown")}</h2>
            <pre class="err">${esc(resp.error)}</pre>`;
  }
  let html = `<h1>Summary</h1><p class="summary">${esc(resp.summary)}</p>`;
  const t = resp.trace;
  if (!t) return html;

  const badge = (b: boolean, label: string) =>
    b ? `<span class="badge">${label}</span>` : "";
  html += `<details open><summary>① Pivot translation
    ${badge(t.translation.repaired, "repaired")}
    ${badge(t.translation.fell_back, "fell back")}
    <span class="score">score ${t.translation.selected_score.toFixed(3)}</span>
    </summary><pre>${esc(t.translation.pivot_code)}</pre></details>`;

  const exs = t.retrieved.map(
    (e) => `<li><b>[${e.score.toFixed(2)}]</b> ${esc(e.summary)}
            <pre>${esc(e.code)}</pre></li>`).join("");
  html += `<details><summary>② Retrieved examples (${t.retrieved.length})</summary>
           <ul>${exs}</ul></details>`;

  const blocks = t.core_blocks.map(
    (b) => `<li>[${b.block_type} ${b.prob.toFixed(2)}] ${esc(b.text)}</li>`).join("");
  html += `<details><summary>③ Core statement blocks (${t.core_blocks.length})</summary>
           <ul class="blocks">${blocks}</ul></details>`;

  html += `<details><summary>④ Final prompt</summary>
           <pre>${esc(t.prompt)}</pre></details>`;
  return html;
}

const STYLE = `
  body { font-family: var(--vscode-font-family); padding: 12px; }
  .summary { font-size: 1.1em; font-weight: 600; }
  pre { background: var(--vscode-textCodeBlock-background); padding: 8px;
        white-space: pre-wrap; border-radius: 4px; }
  .badge { background: var(--vscode-badge-background);
           color: var(--vscode-badge-foreground); border-radius: 4px;
           padding: 0 6px; margin-left: 6px; font-size: 0.8em; }
  .score { color: var(--vscode-descriptionForeground); margin-left: 6px; }
  .err { color: var(--vscode-errorForeground); }
  details { margin-top: 10px; } summary { cursor: pointer; font-weight: 600; }
`;

export class ResultPanel {
  private static current: vscode.WebviewPanel | undefined;

  static show(resp: SummarizeResponse) {
    const col = vscode.ViewColumn.Beside;
    if (!this.current) {
      this.current = vscode.window.createWebviewPanel(
        "codeSummaryResult", "Code Summary", col, { enableScripts: false });
      this.current.onDidDispose(() => (this.current = undefined));
    }
    this.current.webview.html =
      `<!DOCTYPE html><html><head><style>${STYLE}</style></head>
       <body>${renderBody(resp)}</body></html>`;
    this.current.reveal(col);
  }

  static loading() {
    if (!this.current) {
      this.current = vscode.window.createWebviewPanel(
        "codeSummaryResult", "Code Summary", vscode.ViewColumn.Beside,
        { enableScripts: false });
      this.current.onDidDispose(() => (this.current = undefined));
    }
    this.current.webview.html =
      `<!DOCTYPE html><html><head><style>${STYLE}</style></head>
       <body><p>Summarizing…</p></body></html>`;
  }
}
```

- [ ] **Step 2: Type-check**

Run: `cd extension && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add extension/src/panel.ts
git commit -m "feat: webview panel rendering summary + collapsible trace"
```

---

## Task 16: Extension entrypoint (command, status bar, backend lifecycle)

**Files:**
- Create: `extension/src/extension.ts`

- [ ] **Step 1: Write `extension/src/extension.ts`**

```ts
import * as vscode from "vscode";
import { ChildProcess, spawn } from "child_process";
import { RawSettings, buildRequest } from "./config";
import { getHealth, postSummarize } from "./client";
import { ResultPanel } from "./panel";

let backendProc: ChildProcess | undefined;
let statusItem: vscode.StatusBarItem;

function readSettings(): RawSettings {
  const c = vscode.workspace.getConfiguration("codeSummary");
  return {
    mode: c.get("mode", "online") as "online" | "offline",
    online: {
      baseUrl: c.get("online.baseUrl", "https://api.openai.com/v1"),
      apiKey: c.get("online.apiKey", ""),
      model: c.get("online.model", "gpt-4o-mini"),
    },
    offline: {
      baseUrl: c.get("offline.baseUrl", "http://localhost:8080/v1"),
      model: c.get("offline.model", "local-model"),
    },
    params: {
      k: c.get("params.k", 5),
      temperatures: c.get("params.temperatures", [0, 0.4, 0.8]),
      lambda: c.get("params.lambda", 0.5),
      threshold: c.get("params.threshold", 0.5),
      maxRepairIters: c.get("params.maxRepairIters", 3),
    },
  };
}

function backendUrl(): string {
  return vscode.workspace.getConfiguration("codeSummary").get(
    "backend.url", "http://localhost:8000");
}

function updateStatus() {
  const s = readSettings();
  const icon = s.mode === "offline" ? "$(vm)" : "$(cloud)";
  const model = s.mode === "offline" ? s.offline.model : s.online.model;
  statusItem.text = `${icon} Summary: ${s.mode} · ${model}`;
  statusItem.show();
}

async function ensureBackend(): Promise<boolean> {
  const url = backendUrl();
  if (await getHealth(url)) return true;
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  if (!cfg.get("backend.autoStart", true)) {
    vscode.window.showErrorMessage(
      `Code Summary backend not reachable at ${url}. Start it manually.`);
    return false;
  }
  startBackend();
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 500));
    if (await getHealth(url)) return true;
  }
  vscode.window.showErrorMessage("Backend did not become healthy in time.");
  return false;
}

function startBackend() {
  if (backendProc) return;
  const cfg = vscode.workspace.getConfiguration("codeSummary");
  const python = cfg.get("backend.pythonPath", "python");
  const cwd = cfg.get("backend.cwd", "") ||
    (vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd());
  backendProc = spawn(
    python, ["-m", "uvicorn", "app.main:app", "--port", "8000"],
    { cwd, env: process.env });
  backendProc.stderr?.on("data", (d) => console.log(`[backend] ${d}`));
  backendProc.on("exit", () => (backendProc = undefined));
}

function stopBackend() {
  backendProc?.kill();
  backendProc = undefined;
}

async function summarizeSelection() {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.selection.isEmpty) {
    vscode.window.showWarningMessage("Select some code first.");
    return;
  }
  const code = editor.document.getText(editor.selection);
  const language = editor.document.languageId;
  if (!(await ensureBackend())) return;

  ResultPanel.loading();
  try {
    const body = buildRequest(code, language, readSettings());
    const resp = await postSummarize(backendUrl(), body);
    ResultPanel.show(resp);
  } catch (e: any) {
    vscode.window.showErrorMessage(`Summarize failed: ${e.message}`);
  }
}

export function activate(ctx: vscode.ExtensionContext) {
  statusItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right, 100);
  statusItem.command = "codeSummary.toggleMode";
  updateStatus();

  ctx.subscriptions.push(
    statusItem,
    vscode.commands.registerCommand("codeSummary.summarizeSelection", summarizeSelection),
    vscode.commands.registerCommand("codeSummary.startBackend", startBackend),
    vscode.commands.registerCommand("codeSummary.stopBackend", stopBackend),
    vscode.commands.registerCommand("codeSummary.toggleMode", async () => {
      const cfg = vscode.workspace.getConfiguration("codeSummary");
      const next = cfg.get("mode", "online") === "online" ? "offline" : "online";
      await cfg.update("mode", next, vscode.ConfigurationTarget.Global);
      updateStatus();
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("codeSummary")) updateStatus();
    }),
  );
}

export function deactivate() {
  stopBackend();
}
```

- [ ] **Step 2: Build the extension**

Run: `cd extension && npm run build && npx tsc --noEmit`
Expected: `dist/extension.js` produced, no type errors.

- [ ] **Step 3: Commit**

```bash
git add extension/src/extension.ts
git commit -m "feat: extension activation, command, status bar, backend lifecycle"
```

---

## Task 17: End-to-end smoke + README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Low-Resource Code Summary (paper artifact)

VS Code thin client + FastAPI backend implementing the paper's pivot-retrieval
4-stage pipeline. Components ship as stubs so the shell runs end-to-end; plug
real implementations into `backend/app/components/*.py`.

## Backend
```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -v
python -m uvicorn app.main:app --port 8000
```

## CLI (reproducible run)
```bash
python cli.py --code-file snippet.rb --language ruby \
  --base-url http://localhost:8080/v1 --model local-model
```

## Extension
```bash
cd extension && npm install && npm run build
# Press F5 in VS Code to launch the Extension Development Host.
```
Select code → right-click → "Code Summary: Summarize Selection". Toggle
online/offline from the status bar item.

## Online vs offline
Single OpenAI-compatible client; only `base_url` differs.
- Online: `https://api.openai.com/v1` (or DeepSeek, etc.)
- Offline: `http://localhost:8080/v1` (llama.cpp `llama-server`)
````

- [ ] **Step 2: Manual end-to-end smoke**

1. Start backend: `cd backend && . .venv/bin/activate && python -m uvicorn app.main:app --port 8000`
2. `curl -s localhost:8000/health` → expect `{"status":"ok",...}`.
3. With a real local model running (`llama-server -m model.gguf --port 8080`), run:
```bash
curl -s localhost:8000/summarize -H 'Content-Type: application/json' -d '{
  "code":"def foo; 1; end","language":"ruby",
  "model":{"base_url":"http://localhost:8080/v1","api_key":"sk-no-key","model":"local-model"},
  "trace":true}' | python -m json.tool
```
Expected: JSON with non-empty `summary` and a populated `trace`.
4. In VS Code (F5 host): select code, run the command, confirm the panel shows the summary + 4 trace sections, and the status bar toggles online/offline.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with backend, CLI, and extension run instructions"
```

---

## Plugging in real implementations (post-shell)

Each component's constructor + method signatures are final. Replace stub bodies in place:
- `translator.py` — multi-temperature sampling, AST validate/repair, back-translation selection (BLEU + SBERT via `self.embedder`).
- `retriever.py` — build BM25 index from `corpus_path` (set `CS_CORPUS_PATH`); query in `retrieve`.
- `extractor.py` — load classifier from `weights_path` (set `CS_EXTRACTOR_WEIGHTS`); AST semantic split + scoring + threshold.
- `embedder.py` — set `CS_LOAD_SBERT=1` to load the real SBERT model.
- `generator.py` — already real; adjust prompt template / tags to match Appendix A.2 if needed.

No API, pipeline, extension, or schema changes are required to swap in real logic.
```
```
```

---

## Self-Review

**Spec coverage:** §2 architecture → Tasks 1,10,16. §3 component interfaces → Tasks 5-9 (signatures match spec exactly). §4 API contract → Task 10 (+ Params alias `lambda`). §5 extension (commands, status bar, panel, config, lifecycle) → Tasks 12,15,16. §6 error handling (per-stage `failed_stage`, fall-back not an error) → Task 9. §7 testing (FakeLLMClient, stub-first) → Tasks 3,9,10,13. §8 tech stack → Task 1,12. §9 YAGNI (no training, no SSE, VS Code only) → respected.

**Type consistency:** `TranslationResult`, `CoreBlock`, `Example`, `Trace`, `SummarizeResponse` field names identical across schemas (Task 2), pipeline (Task 9), generator (Task 8), TS client interfaces (Task 14), and panel (Task 15). `make_llm` overridden in tests (Tasks 10,11) matches definitions. `buildRequest`/`activeModel` names consistent (Tasks 13,16).

**Placeholders:** none — every code step contains full content.
