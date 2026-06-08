from __future__ import annotations
import re
from app.schemas import Example, CoreBlock
from app.models.llm import LLMClient

_SYSTEM = (
    "You are an expert programmer. Generate a concise one-sentence natural-"
    "language summary of the given code. The provided core statement blocks "
    "highlight the key behavior. Output ONLY the summary wrapped in "
    "<summary></summary> tags."
)
_INSTRUCTION = "Summarize what the code does in one sentence."
_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def _blocks_text(blocks: list[CoreBlock]) -> str:
    return "\n".join(b.text for b in blocks)


class Generator:
    """Structure-guided summary generation with tagged boundaries (real impl)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _build_prompt(self, pivot_code: str, core_blocks: list[CoreBlock],
                      examples: list[Example]) -> str:
        parts: list[str] = []
        for ex in examples:
            parts.append(
                f"<code>\n{ex.code}\n</code>\n"
                f"<core>\n{_blocks_text(ex.core_blocks)}\n</core>\n"
                f"<instruction>{_INSTRUCTION}</instruction>\n"
                f"<summary>{ex.summary}</summary>"
            )
        parts.append(
            f"<code>\n{pivot_code}\n</code>\n"
            f"<core>\n{_blocks_text(core_blocks)}\n</core>\n"
            f"<instruction>{_INSTRUCTION}</instruction>\n"
            f"<summary>"
        )
        return "\n\n".join(parts)

    def generate(self, pivot_code: str, core_blocks: list[CoreBlock],
                 examples: list[Example]) -> tuple[str, str]:
        prompt = self._build_prompt(pivot_code, core_blocks, examples)
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        raw = self.llm.chat(messages, temperature=0.0)
        m = _TAG_RE.search(raw)
        summary = m.group(1).strip() if m else raw.strip()
        return summary, prompt
