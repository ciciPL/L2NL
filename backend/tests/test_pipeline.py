from app.pipeline import Pipeline
from app.components.translator import Translator
from app.components.retriever import Retriever
from app.components.extractor import Extractor
from app.components.generator import Generator
from app.models.llm import FakeLLMClient
from app.models.embedder import Embedder
from app.schemas import Params

# Single forward temperature keeps the FakeLLMClient response order predictable:
# call 1 = translation (forward), call 2 = summary generation.
P = Params(temperatures=[0.0])
PIVOT = "<PYTHON>\ndef foo():\n    return 1\n</PYTHON>"


def _pipeline(llm):
    return Pipeline(
        translator=Translator(llm, Embedder(), P),
        retriever=Retriever(),
        extractor=Extractor(),
        generator=Generator(llm),
    )


def test_run_returns_summary_and_full_trace():
    llm = FakeLLMClient(responses=[PIVOT, "It does X."])
    resp = _pipeline(llm).run("def foo(): return 1", "ruby", P, trace=True)
    assert resp.error is None
    assert resp.summary == "It does X."
    assert resp.trace is not None
    assert resp.trace.translation.pivot_code == "def foo():\n    return 1"
    assert resp.trace.translation.fell_back is False
    assert len(resp.trace.retrieved) == P.k
    assert "# [USER INPUT CODE]" in resp.trace.prompt


def test_run_without_trace_omits_trace():
    llm = FakeLLMClient(responses=[PIVOT, "Y."])
    resp = _pipeline(llm).run("x", "ruby", P, trace=False)
    assert resp.summary == "Y."
    assert resp.trace is None


def test_stage_failure_is_reported():
    class BoomRetriever(Retriever):
        def retrieve(self, pivot_code, k):
            raise RuntimeError("index missing")

    llm = FakeLLMClient(responses=[PIVOT])
    pipe = Pipeline(
        translator=Translator(llm, Embedder(), P),
        retriever=BoomRetriever(),
        extractor=Extractor(),
        generator=Generator(llm),
    )
    resp = pipe.run("x", "ruby", P, trace=True)
    assert resp.error is not None
    assert resp.failed_stage == "retrieve"
    assert "index missing" in resp.error
