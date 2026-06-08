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
