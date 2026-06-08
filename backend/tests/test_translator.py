from app.components.translator import Translator, extract_clean_code, validate_syntax
from app.models.llm import FakeLLMClient
from app.models.embedder import Embedder
from app.schemas import Params

VALID = "<PYTHON>\ndef f():\n    return 1\n</PYTHON>"
INVALID = "<PYTHON>\ndef f( :\n</PYTHON>"


def _tr(llm):
    return Translator(llm, Embedder(), Params(temperatures=[0.0], max_repair_iters=2))


def test_extract_and_validate_helpers():
    assert extract_clean_code(VALID) == "def f():\n    return 1"
    ok, _, _ = validate_syntax("def f():\n    return 1")
    assert ok is True
    bad, msg, _ = validate_syntax("def f( :")
    assert bad is False and msg


def test_valid_on_first_try():
    tr = _tr(FakeLLMClient(responses=[VALID]))
    res = tr.translate("def f; 1; end", "ruby")
    assert res.pivot_code == "def f():\n    return 1"
    assert res.repaired is False
    assert res.fell_back is False
    assert res.candidates == ["def f():\n    return 1"]


def test_repair_recovers_invalid_candidate():
    # forward call returns invalid; repair call returns valid
    tr = _tr(FakeLLMClient(responses=[INVALID, VALID]))
    res = tr.translate("def f; 1; end", "ruby")
    assert res.pivot_code == "def f():\n    return 1"
    assert res.repaired is True
    assert res.fell_back is False


def test_falls_back_to_tau0_when_all_fail():
    tr = Translator(FakeLLMClient(responses=[INVALID, INVALID]),
                    Embedder(), Params(temperatures=[0.0], max_repair_iters=1))
    res = tr.translate("def f; 1; end", "ruby")
    assert res.fell_back is True
    assert res.pivot_code == "def f( :"   # tau=0 candidate kept despite invalid
