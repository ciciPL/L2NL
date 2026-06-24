from __future__ import annotations
import ast
import re
from app.schemas import TranslationResult, Params
from app.models.llm import LLMClient
from app.models.embedder import Embedder

# Ported from L2NL_release/02_translation/translate_stepBystep_vllm.py (paper §3.2).
# Single-snippet adaptation: multi-temperature forward generation -> AST validation
# -> parser-feedback iterative repair. LLM calls route through LLMClient instead of
# vllm. Back-translation candidate selection (find_bestCode, BLEU + Jina embedding)
# is NOT yet wired — it needs a real embedder; see _select_best TODO. For now the
# paper's stable fallback (greedy tau=0 candidate) is used.

_REPAIR_TEMPERATURE = 0.2
_STOP = ["<|endoftext|>", "<|im_end|>", "<|EOT|>"]


def validate_syntax(code: str):
    if not code:
        return False, "Empty code", "Empty"
    try:
        ast.parse(code)
        return True, None, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}", "SyntaxError"
    except Exception as e:  # noqa: BLE001
        return False, str(e), type(e).__name__


def extract_clean_code(generated_text: str) -> str:
    xml_match = re.search(r"<PYTHON(?:\s*>|\s+)(.*?)</PYTHON>", generated_text,
                          re.DOTALL | re.IGNORECASE)
    if xml_match:
        content = xml_match.group(1).strip()
        md_match = re.search(r"```(?:python)?\s*(.*?)\s*```", content,
                             re.DOTALL | re.IGNORECASE)
        return md_match.group(1).strip() if md_match else content

    md_match = re.search(r"```python\s*(.*?)\s*```", generated_text,
                         re.DOTALL | re.IGNORECASE)
    if md_match:
        return md_match.group(1).strip()

    clean_text = generated_text
    for pattern in [r"\nNote:", r"\nExplanation:", r"The Python code"]:
        parts = re.split(pattern, clean_text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1:
            clean_text = parts[0]
    return clean_text.strip()


def format_prompt_translate(source_code: str, lang: str) -> str:
    return f"""You are an expert code translator. Translate the following {lang} code to Python.

Requirements:
1. Preserve the original logic and functionality
2. Use Python idioms and best practices
3. CRITICAL: Output ONLY the Python code logic wrapped inside <PYTHON> and </PYTHON> tags.
4. Do NOT use markdown code blocks (like ```python).
5. Do NOT output explanations or notes or testcase.

Example:

Input:
func add(a, b) {{ return a + b }}

Output:
<PYTHON>
def add(a, b):
    return a + b
</PYTHON>

Real Task:

{lang} Input:
{source_code}

Output:"""


def format_prompt_repair(source_code: str, failed_candidate: str,
                         error_log: str, lang: str) -> str:
    return f"""You are an expert code debugger.
Source ({lang}):
{source_code}

Your previous Python translation:
{failed_candidate}

Error Message:
{error_log}

Fix the Python code based on the error. Return ONLY the fixed code.
Python Code:"""


class Translator:
    """Cross-language translation to a Python pivot (paper §3.2)."""

    def __init__(self, llm: LLMClient, embedder: Embedder, params: Params):
        self.llm = llm
        self.embedder = embedder
        self.params = params

    def _forward(self, code: str, src_lang: str) -> list[dict]:
        """One candidate per configured temperature."""
        candidates: list[dict] = []
        for temp in self.params.temperatures:
            messages = [
                {"role": "system", "content": "You are an expert code translator."},
                {"role": "user", "content": format_prompt_translate(code, src_lang)},
            ]
            raw = self.llm.chat(messages, temperature=temp, stop=_STOP)
            clean = extract_clean_code(raw)
            ok, err, _ = validate_syntax(clean)
            candidates.append({"code": clean, "temperature": temp,
                               "valid": ok, "error": err, "repaired": False})
        return candidates

    def _repair(self, code: str, src_lang: str, candidates: list[dict]) -> None:
        """Iterative AST-feedback repair; appends repaired candidates in place."""
        for _ in range(self.params.max_repair_iters):
            if any(c["valid"] for c in candidates):
                return
            invalids = sorted((c for c in candidates if not c["valid"]),
                              key=lambda c: len(c["code"]) if c["code"] else 9999)
            base = invalids[0]
            prompt = format_prompt_repair(code, base["code"], base["error"] or "", src_lang)
            raw = self.llm.chat([{"role": "user", "content": prompt}],
                                temperature=_REPAIR_TEMPERATURE, stop=_STOP)
            new_code = extract_clean_code(raw)
            ok, err, _ = validate_syntax(new_code)
            candidates.append({"code": new_code, "temperature": _REPAIR_TEMPERATURE,
                               "valid": ok, "error": err, "repaired": True})

    @staticmethod
    def _tau0(candidates: list[dict]) -> dict:
        for c in candidates:
            if c["temperature"] == 0.0:
                return c
        return candidates[0]

    def _select_best(self, candidates: list[dict]) -> tuple[dict, bool]:
        """Pick the final pivot.

        TODO: faithful selection back-translates each valid candidate and scores
        S = lambda*BLEU + (1-lambda)*SBERT_sim (paper §3.2, find_bestCode). That
        needs a real embedder; wire here once available. Until then: prefer the
        greedy tau=0 candidate if valid, else the first valid one; if none valid,
        fall back to the tau=0 candidate (paper's stable fallback).
        """
        valid = [c for c in candidates if c["valid"]]
        if not valid:
            return self._tau0(candidates), True
        tau0 = self._tau0(candidates)
        return (tau0 if tau0["valid"] else valid[0]), False

    def translate(self, code: str, src_lang: str) -> TranslationResult:
        if src_lang.lower() in {"python", "py"}:
            return TranslationResult(
                pivot_code=code.strip(),
                candidates=[code.strip()],
                selected_score=None,
                repaired=False,
                fell_back=False,
            )
        candidates = self._forward(code, src_lang)
        self._repair(code, src_lang, candidates)
        chosen, fell_back = self._select_best(candidates)
        return TranslationResult(
            pivot_code=chosen["code"],
            candidates=[c["code"] for c in candidates],
            selected_score=None,
            repaired=chosen["repaired"],
            fell_back=fell_back,
        )
