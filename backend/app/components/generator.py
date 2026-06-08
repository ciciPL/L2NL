from __future__ import annotations
from app.schemas import Example, CoreBlock
from app.models.llm import LLMClient

# Ported from L2NL_release/05_summary_gen/finalScript_genNL.py (paper Appendix A.2,
# "full" variant: USE_SENTENCES = USE_FEW_SHOT = USE_DYNAMIC_BM25_SHOTS = True).
# Research code is a single completion-style string for vllm.generate; here it is
# adapted to a chat client (guidelines -> system, examples + target -> user) with
# "</SUMMARY>" as the stop sequence.

_SYSTEM = (
    "You're a specialized AI assisting with Python code summaries, deeply "
    "knowledgeable in computer science.\n\n"
    "GUIDELINES:\n"
    "1. Code Analysis: Read the source code under '# [USER INPUT CODE]'.\n"
    "2. Trace Usage: The '# [KEY LOGIC TRACE]' section provides execution hints. "
    "Use it as an auxiliary aid to understand the core functionality, but avoid "
    "describing low-level implementation details (e.g., specific variable "
    "manipulations) in your summary. Focus on the high-level intent.\n"
    "3. Output Format: Output your natural language summary inside <SUMMARY> and "
    "</SUMMARY> tags.\n"
)
_STOP = ["</SUMMARY>"]


def _format_block_annotation(code: str, blocks: list[CoreBlock],
                             summary: str | None, return_parts: bool):
    """Port of format_block_annotation_style (USE_SENTENCES=True)."""
    text = "# ==================================================\n"
    text += "# [USER INPUT CODE]\n"
    text += f"{code.strip()}\n\n"

    formatted = [f"# > {b.text.strip()}" for b in blocks if b.text.strip()]
    trace_content = "\n".join(formatted)
    text += "# [KEY LOGIC TRACE]\n"
    text += (f"{trace_content}\n\n" if trace_content else "\n")

    text += "Concisely summarize the code provided in 1-3 sentences.\n"

    assistant_text = f"<SUMMARY>\n{summary.strip()}\n</SUMMARY>" if summary else ""
    if return_parts:
        return text, assistant_text
    return text + (assistant_text + "\n" if assistant_text else "")


def _clean_extracted_summary(raw_text: str) -> str:
    """Port of clean_extracted_summary."""
    content = raw_text.split("<SUMMARY>")[-1] if "<SUMMARY>" in raw_text else raw_text
    content = content.replace("</SUMMARY>", "").strip()
    if content.startswith("#"):
        content = content.lstrip("#").strip()
    return content


class Generator:
    """Structure-guided summary generation with tagged boundaries.

    Ported from the paper's finalScript_genNL.py (full variant). The pivot code's
    core blocks form the [KEY LOGIC TRACE]; retrieved examples become few-shot
    demonstrations carrying their own code/trace/summary.
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _build_user_text(self, pivot_code: str, core_blocks: list[CoreBlock],
                         examples: list[Example]) -> str:
        text = "Examples:\n\n"
        for ex in examples:
            user_part, assistant_part = _format_block_annotation(
                ex.code, ex.core_blocks, ex.summary, return_parts=True)
            text += user_part.strip() + "\n"
            text += assistant_part.strip() + "\n\n"

        target_user_part, _ = _format_block_annotation(
            pivot_code, core_blocks, summary=None, return_parts=True)
        text += target_user_part.strip() + "\n"
        text += "<SUMMARY>\n"
        return text

    def generate(self, pivot_code: str, core_blocks: list[CoreBlock],
                 examples: list[Example]) -> tuple[str, str]:
        user_text = self._build_user_text(pivot_code, core_blocks, examples)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_text},
        ]
        raw = self.llm.chat(messages, temperature=0.0, stop=_STOP)
        summary = _clean_extracted_summary(raw)
        full_prompt = f"{_SYSTEM}\n\n{user_text}"
        return summary, full_prompt
