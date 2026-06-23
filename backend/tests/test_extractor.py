import inspect

from app.components import _py_split
from app.components.extractor import build_seq_items, Extractor

CODE = (
    "def f(x):\n"
    "    total = 0\n"
    "    for i in range(x):\n"
    "        total += i\n"
    "    if total > 10:\n"
    "        return total\n"
    "    return 0"
)


def test_build_seq_items_categorizes_by_structure():
    items = build_seq_items(CODE)
    types = [t for _, _, t in items]
    assert types[0] == "signature"          # function_def comes first
    assert "loop" in types
    assert "conditional" in types
    assert "assignment" in types
    # cleaned seq is whitespace-normalized + lowercased; raw keeps original text
    raws = [r for r, _, _ in items]
    cleaned = [c for _, c, _ in items]
    assert any("for i in range(x)" in r for r in raws)
    assert all(c == c.lower() for c in cleaned)


def test_build_seq_items_uses_research_new_python_splitter():
    assert hasattr(_py_split, "split_python_by_structure_new")
    assert "split_python_by_structure_new" in inspect.getsource(build_seq_items)


def test_build_seq_items_recurses_into_nested_python_functions():
    items = build_seq_items(
        "def outer():\n"
        "    def inner():\n"
        "        value = 1\n"
        "        return value\n"
        "    return inner()"
    )
    raws = [r for r, _, _ in items]
    assert "def outer():" in raws
    assert "def inner():" in raws
    assert "value = 1" in raws
    assert not any(r.startswith("def inner():\n") for r in raws)


def test_build_seq_items_handles_async_functions_and_class_methods():
    async_items = build_seq_items("async def fetch():\n    value = 1\n    return value")
    async_raws = [r for r, _, _ in async_items]
    assert "async def fetch():" in async_raws
    assert "value = 1" in async_raws
    assert not any(r.startswith("async def fetch():\n") for r in async_raws)

    class_items = build_seq_items(
        "class Service:\n"
        "    def run(self):\n"
        "        value = 1\n"
        "        return value"
    )
    class_raws = [r for r, _, _ in class_items]
    assert "def run(self):" in class_raws
    assert "value = 1" in class_raws
    assert not any(r.startswith("class Service:\n") for r in class_raws)


def test_build_seq_items_empty_on_syntax_error():
    assert build_seq_items("def f( :") == []


def test_extractor_fallback_without_weights_only_when_stubs_allowed():
    blocks = Extractor(allow_stubs=True).extract("a = 1\n\nb = 2", 0.5)
    assert [b.text for b in blocks] == ["a = 1", "b = 2"]


def test_extractor_without_weights_fails_in_production_mode():
    ex = Extractor()
    try:
        ex.extract("a = 1", 0.5)
    except RuntimeError as e:
        assert "CS_EXTRACTOR_WEIGHTS" in str(e)
    else:
        raise AssertionError("production extractor should fail without weights")
