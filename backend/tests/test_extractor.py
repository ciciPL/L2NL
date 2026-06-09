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


def test_build_seq_items_empty_on_syntax_error():
    assert build_seq_items("def f( :") == []


def test_extractor_fallback_without_weights():
    # No checkpoint configured -> naive line-split fallback (keeps shell runnable).
    blocks = Extractor().extract("a = 1\n\nb = 2", 0.5)
    assert [b.text for b in blocks] == ["a = 1", "b = 2"]
