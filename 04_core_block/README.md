# Core Statement-Block Extraction (Paper §3.4)

This module trains a binary classifier that, for every AST-aligned semantic block of a function,
predicts whether the block belongs to the **core statement set** that should be exposed to the
downstream summary-generation LLM. The classifier is trained with **ROUGE-L greedy labels**
against the reference summary (see paper §3.4 for the labeling rule).

## Files

| File | Purpose | Self-contained? |
|---|---|---|
| `make_dataset_label.py` | Build the per-block structural input: for every record, run `split_python_by_structure_new` to AST-split the *translated Python pivot code* (and each retrieved candidate) into `function_def / loops / conditionals / assignments / others`. Output JSONL feeds `classfier_BM25.py`. | Depends on `utils.py` (`from utils import *`) |
| `classfier_BM25.py` | **Inference driver** for the binary classifier. Encoder is **CodeBERT** (`microsoft/codebert-base`); a two-level hierarchical head (word-level + statement-level) produces a per-block probability. Loads a pre-trained checkpoint and writes per-block predictions to a new JSONL. | Requires `model.py::SelectorNet` from the EACS repo (see below) + a checkpoint you trained yourself with the EACS training script |
| `utils.py` | AST-aligned block splitters for **5 LRPLs + Python pivot + Ruby**: `split_julia_by_structure`, `split_lua_by_structure`, `split_ocaml_by_structure`, `split_r_by_structure`, `split_racket_by_structure`, `split_python_by_structure_new`, `split_ruby_by_structure`. Used by `make_dataset_label.py`. | — |

## On `split_python_by_structure_new`

Why "new": the LRPL-to-Python translation step (§3.2) often produces **nested function
definitions** (the LRPL source had an outer wrapper, or the translation introduced an inner
helper). The original `split_python_by_structure` only handled the outermost function. The new
splitter recurses through every nested `ast.FunctionDef`:

- `function_def` is returned as a **list** (one signature per nested function),
- the body of every nested function is recursively decomposed into
  `loops / conditionals / assignments / others`.

This is the splitter `make_dataset_label.py` actually invokes; the other per-language splitters
in `utils.py` exist so the same pipeline can be re-run on raw LRPL code if needed.

## Relation to Prior Work — EACS

The classifier framework and the "extract-then-summarize" paradigm follow **EACS**
(Sun et al., 2024, [paper ref19] of this manuscript):

> **EACS:** Sun, W. et al. *An Extractive-and-Abstractive Framework for Source Code Summarization.*
> ACM TOSEM 33(3), 2024.
> **GitHub:** https://github.com/wssun/EACS

What we keep from EACS:
- The two-stage architecture (extractive labeling → classifier training → use as input to a generator).
- The CodeBERT-based encoder for sentence-level binary classification.
- ROUGE-L–based greedy labeling rule for constructing training data.

What we change from EACS (paper §3.4):
- **AST-aligned semantic-block splitting** instead of EACS's fine-grained lexical tokenization.
  Each block is a complete loop / conditional / assignment / function-signature unit, preserving
  control-flow and contextual information rather than splitting tokens linearly.
- **Nested-function-aware splitter** (`split_python_by_structure_new`) added for the pivot-Python
  case where LRPL translations often introduce inner functions.
- Per-language structural splitters for Julia / Lua / OCaml / R / Racket (in `utils.py`),
  enabling the same pipeline to also process raw LRPL code when needed.

`utils.py` is a customised superset of EACS's helper module; the EACS-original token-level
helpers (`identifier_splitting.py`, `rouge_not_a_wrapper.py`) are **not required** by the
cleaned pipeline because they were only used by token-level training-data builders that we no
longer call. If you want to rerun the original EACS workflow for comparison, clone the EACS
repo above and use its scripts directly.

### Files you need from the EACS repo

`classfier_BM25.py` is the inference half of the pipeline (load checkpoint → predict per-block
labels → write JSONL). The classifier *definition* and the training loop are inherited from
EACS unchanged, so we do not redistribute them here. Before running `classfier_BM25.py`, clone
EACS and copy/import two pieces:

| What | Where in EACS | Where to put it |
|---|---|---|
| `SelectorNet` class (the two-level hierarchical CodeBERT head) | `model.py` | `04_core_block/model.py` (next to `classfier_BM25.py`) |
| Training loop that produces `checkpoint-best-loss/pytorch_model.bin` | EACS's `run.py` / equivalent | run inside the EACS repo against the PCSD splits; copy the resulting checkpoint to the path passed via `--load_model_path` |

Without `model.py`, `classfier_BM25.py` will raise an explicit `ImportError` at start-up.

## Quick Start

```bash
# Step 1: build structural input (assumes upstream retrieval has produced a JSONL with
# `best_python_code` + `retrieved_candidates`)
python make_dataset_label.py \
    --input  ./data/CSN/ruby/trans_qwen3th/test_retrieved.jsonl \
    --output ./data/CSN/ruby/trans_qwen3th/test_sentences.jsonl

# Step 2: train / predict with the classifier (defaults to microsoft/codebert-base)
python classfier_BM25.py
#   - Loads CodeBERT, fine-tunes hierarchical word+statement heads
#   - Saves best checkpoint to ./output_structure/<lang>/checkpoint-best-loss/
#   - Predicts on the file in --predict_file and writes <lang>_test_sentences_preds.jsonl
```

## Datasets needed

- **PCSD** (Python Code-Summary Dataset) — used to train the classifier.
  Place AST-clean splits under `./data/Clean_PCSD-ast/{train,valid,test}/*.jsonl`.
- **CodeSearchNet (Ruby)** + the pivot-translated `*_retrieved.jsonl` produced by the upstream
  translation / retrieval pipeline (`02_translation/` + `03_retrieval/BM25.py`).

## Notes

- The script name `classfier_BM25.py` is historical (kept for traceability with the original
  experiment logs); the classifier itself does *not* use BM25 — the encoder is CodeBERT.
- Hyper-parameters (`max_word_length=32`, `max_stat_length=32`, `word_hidden_size=128`,
  `stat_hidden_size=256`, `embed_size=768`) match the paper's §4.3.
