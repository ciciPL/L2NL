from app.components.generator import Generator
from app.models.llm import FakeLLMClient
from app.schemas import Example, CoreBlock


def test_generate_cleans_summary_and_builds_paper_prompt():
    # With stop="</SUMMARY>" the model returns just the summary body.
    llm = FakeLLMClient(responses=["It adds two numbers."])
    gen = Generator(llm)
    examples = [Example(code="def add(a,b): return a+b",
                        core_blocks=[CoreBlock(text="return a+b", block_type="other", prob=1.0)],
                        summary="Adds two numbers.", score=1.0)]
    blocks = [CoreBlock(text="return a+b", block_type="other", prob=1.0)]
    summary, prompt = gen.generate("def add(a,b): return a+b", blocks, examples)

    assert summary == "It adds two numbers."
    # Paper prompt structure present
    assert "# [USER INPUT CODE]" in prompt
    assert "# [KEY LOGIC TRACE]" in prompt
    assert "# > return a+b" in prompt          # core block rendered as trace line
    assert "<SUMMARY>\nAdds two numbers.\n</SUMMARY>" in prompt  # few-shot answer
    assert prompt.rstrip().endswith("<SUMMARY>")  # target left open for completion
    # stop sequence is passed through
    assert llm.calls[0]["stop"] == ["</SUMMARY>"]


def test_generate_strips_echoed_tags():
    llm = FakeLLMClient(responses=["noise <SUMMARY>\nClean body.\n</SUMMARY>"])
    summary, _ = Generator(llm).generate("code", [], [])
    assert summary == "Clean body."
