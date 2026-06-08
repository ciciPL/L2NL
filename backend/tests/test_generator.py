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
