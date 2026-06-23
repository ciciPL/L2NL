import json
from app.components.retriever import Retriever, clean_python_code, makestr


def test_clean_strips_docstring():
    code = 'def f():\n    """doc here"""\n    return 1'
    cleaned = clean_python_code(code)
    assert "doc here" not in cleaned
    assert "return 1" in cleaned


def test_makestr_joins_tokens():
    assert makestr(["adds", "two", "numbers", "."]) == "adds two numbers."


def test_retriever_without_corpus_returns_canned_only_when_stubs_allowed():
    out = Retriever(allow_stubs=True).retrieve("def f(): pass", 3)
    assert len(out) == 3


def test_retriever_without_corpus_fails_in_production_mode():
    r = Retriever()
    try:
        r.retrieve("def f(): pass", 3)
    except RuntimeError as e:
        assert "CS_CORPUS_PATH" in str(e)
    else:
        raise AssertionError("production retriever should fail without corpus")


def test_retriever_bm25_ranks_relevant_first(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"code": "def add(a, b):\n    return a + b",
         "docstring_tokens": ["adds", "two", "numbers"]},
        {"code": "def read_file(path):\n    return open(path).read()",
         "docstring_tokens": ["reads", "a", "file"]},
    ]
    corpus.write_text("\n".join(json.dumps(r) for r in rows))

    r = Retriever(str(corpus))
    out = r.retrieve("def sum2(x, y):\n    return x + y", 2)
    assert len(out) == 2
    # the add/return example should outrank the file-reading one
    assert out[0].summary == "adds two numbers"
    assert out[0].score >= out[1].score
