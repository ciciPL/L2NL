# Integrating the real L2NL pipeline into the shell

Source: `L2NL_release/` (the paper's research code; batch/offline, vllm-based).
This file maps each research script onto a shell component and records exactly
what is still missing (models / data on the server or external repos).

The shell already runs end-to-end with stubs. "Plugging in" = port each
algorithm into the component's final-signature method and route any LLM call
through the existing `LLMClient` (OpenAI-compatible: vllm / llama-server /
DeepSeek). No API / schema / extension / pipeline changes are required.

## Component mapping

| Shell component | Research source | Reusable logic | Status |
|---|---|---|---|
| `components/translator.py` | `02_translation/translate_stepBystep_vllm.py` (multi-temp forward + AST repair), `translate_back_2_lrpl*.py` (back-translation), `translate_find_bestCode_from_back_vllm.py` (BLEU + Jina selection) | `validate_syntax`, `extract_clean_code`, `format_prompt_translate/repair`, `JinaVLLMScorer` scoring | **DONE** — forward + AST repair ported, LLM via `LLMClient`, verified live on DeepSeek. Back-translation selection still TODO (needs embedder); uses tau=0 fallback. |
| `components/retriever.py` | `03_retrieval/BM25.py` (`run_retrieval`, `clean_python_code`) | whole file (pure `rank_bm25`) | Portable now. **Needs HR corpus data** (CodeSearchNet Python/Ruby) → still stub. |
| `components/extractor.py` | `04_core_block/classfier_BM25.py` (inference driver) + `utils.py` (`split_python_by_structure`, `split_ruby_by_structure`, … per-language AST splitters) | AST splitters in `utils.py` are pure/portable | AST split portable now (pivot is Python); **classifier BLOCKED** on 3 missing pieces (below). |
| `components/generator.py` | `05_summary_gen/finalScript_genNL.py` (`build_completion_prompt`, `format_block_annotation_style`, `clean_extracted_summary`) | prompt construction | **DONE** — paper "full" variant prompt ported, verified live on DeepSeek. |
| `models/embedder.py` | `JinaVLLMScorer` in `02_translation/...` | — | Research uses `jinaai/jina-code-embeddings-1.5b` via vllm. Shell uses sentence-transformers. Pick one (see Open questions). |

## Missing assets (none are in `L2NL_release.zip`)

### Core-block classifier (the server-side blocker) — `components/extractor.py`
1. **`model.py::SelectorNet`** — two-level hierarchical CodeBERT head. *Not
   redistributed in the release*; copy from the EACS repo
   (https://github.com/wssun/EACS). → place next to the ported extractor code.
2. **Trained checkpoint** `checkpoint-best-loss/pytorch_model.bin` — **on the
   user's server.** → will be loaded via env `CS_EXTRACTOR_WEIGHTS` (already
   wired in `config.py`; stub ignores it for now).
3. **`microsoft/codebert-base`** encoder — download from HuggingFace.

### Data (download separately; not models)
- **CodeSearchNet (Ruby/Python)** — HR pivot corpus for BM25 (`retriever.py`) →
  env `CS_CORPUS_PATH` (already wired).
- **MultiPL-T** — Julia/Lua/OCaml/R/Racket LRPL test sets (eval only).
- **PCSD** — only needed to *train* the classifier (not for inference).

### Translator / summarizer LLMs
Research used Qwen2.5-Coder-14B / Llama-3.1-8B / deepseek-coder etc. via vllm.
In the shell these are **not vendored** — any OpenAI-compatible endpoint works
through `LLMClient` (online API or a local vllm/llama-server). No placeholder
needed beyond the model config already in the request.

## Placeholder state today
- `generator.py` → ✅ real (paper full-variant prompt).
- `translator.py` → ✅ real forward + AST repair (LLM via client). Back-translation
  selection TODO in `_select_best` (needs embedder); tau=0 fallback for now.
- `extractor.py` → naive line-split stub. Next non-server step: port the Python
  AST splitter from `utils.py` for real semantic blocks; classifier still needs
  (1)(2)(3). Env `CS_EXTRACTOR_WEIGHTS` reserved.
- `retriever.py` → canned examples; swap in BM25 once corpus path is provided.
  Env `CS_CORPUS_PATH` reserved.
- `embedder.py` → hash-vector stub; set `CS_LOAD_SBERT=1` for real model.

## Open questions before wiring
1. Can you provide EACS `model.py` (SelectorNet), or should we re-derive the
   head from the paper's §3.4 description?
2. Where will the classifier checkpoint live (path on the box running the
   backend), and how big is it?
3. Where is the HR BM25 corpus (CodeSearchNet Python) on disk?
4. Embedder for back-translation selection: keep research's Jina
   (`jina-code-embeddings-1.5b`) or substitute a lighter SBERT?

## Reference copy
Research code currently at: `~/Downloads/L2NL_release/` (unzipped).
Decide whether to vendor a copy under `backend/reference/L2NL_release/` or keep
it external and port on demand.
