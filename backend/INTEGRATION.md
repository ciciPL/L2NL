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

## RESOLVED — asset locations (2026-06-08)

All located. Backend will run ON star (Linux, 2×4090). Extension connects over
Tailscale to `http://100.122.192.123:8000`.

| Asset | Location |
|---|---|
| Translator/Summarizer LLM | DeepSeek API (`deepseek-v4-flash`), via `LLMClient`. Done. |
| BM25 corpus (CodeXGLUE python train) | `star:~/code_sum_rag/external/codexglue_python_train.jsonl` |
| Core-block classifier checkpoint | `myci:F:\PycharmProjects\L2NL\script\EASC\output_structure\python\checkpoint-best-loss\pytorch_model.bin` (301MB) → transfer to star |
| SelectorNet code + AST splitter | vendored in `backend/vendor/easc/` (model.py, utils.py, classfier_BM25.py, ast_split_utils.py, make_dataset_label.py) |
| CodeBERT encoder | `star:~/.cache/huggingface/hub/models--microsoft--codebert-base` |

## Extractor wiring recipe (SelectorNet)

The classifier does NOT split code — it takes an already-split statement list.
1. AST split: `ast_split_utils.py::split_python_by_structure(code)` → list of
   statement strings (`cleaned_seqs`). Pivot is Python, so only the Python
   splitter is needed.
2. Build features: per `classfier_BM25.py::convert_examples_to_features`
   (max_stat_length=32, max_word_length=32, RobertaTokenizer of codebert-base).
3. Model: `model.py::SelectorNet(batch_size, word_embeddings_weight,
   word_hidden_size=128, stat_hidden_size=256, max_word_len=32, max_stat_len=32,
   vocab_size=50265, embed_size=768, num_classes=2, imbalance_loss_fct=True)`;
   word_embeddings_weight from `RobertaModel.from_pretrained(codebert).embeddings
   .word_embeddings.weight`; `load_state_dict(torch.load(checkpoint))`.
4. Forward: `model(source_ids, word_masks, stat_masks, None)` → `(num,
   active_mask, probs)`; `argmax(probs,1)` per statement; label==1 statements are
   the core blocks (→ `CoreBlock`).

Needs `[full]` extra + torch + transformers in the backend env on star.

## Retriever wiring recipe (BM25)
Port `03_retrieval/BM25.py` (`run_retrieval`, `clean_python_code`); build the
rank-bm25 index from `codexglue_python_train.jsonl` at startup; query with the
pivot code.

## Reference copy
`L2NL_release/` unzipped at `/tmp/l2nl_inspect/` (clears on reboot; canonical zip
in `~/Downloads/`). Fuller copy on star at `~/Downloads/cyqProjects/L2NL` and on
myci at `F:\PycharmProjects\L2NL`.
