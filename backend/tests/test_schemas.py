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
