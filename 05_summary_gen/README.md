# Structure-Guided Summary Generation (Paper §3.5 / §4.5 RQ5)

LLM inference for the final NL summary given the pivot Python code (from `02_translation/`),
optional core statement-block predictions (from `04_core_block/`), and optional BM25-retrieved
few-shot examples (from `03_retrieval/`).

## Files

| Script | Purpose |
|---|---|
| **`finalScript_genNL.py`** ⭐ | **Unified entry point for all 4 RQ5 ablation variants** — selected via three CLI toggles. |
| `gen_NL_sentences_vllm_icse_prompt_adapter.py` | ICSE-2025-style prompt structure (Makharev & Ivanov 2025, [ref21]). Used to verify our prompt is competitive with the published ICSE template. |

## RQ5 ablation matrix → CLI toggles

Three flags on `finalScript_genNL.py` select the variant (defaults match **Full**):

```bash
--use_sentences / --no_sentences          # inject [KEY LOGIC TRACE] core-block hints (§3.4)
--use_dynamic_bm25 / --no_dynamic_bm25    # BM25-retrieved dynamic shots vs. static FEW_SHOTS_DATA
--use_few_shot / --no_few_shot            # include few-shot examples at all
```

| Variant | flags | Output file suffix |
|---------|---|:------------------:|
| **Full**   | *(defaults)*                                  | `_nl_full.txt`  |
| w/oKey  | `--no_sentences`                                 | `_nl_woKey.txt` |
| w/oRet  | `--no_dynamic_bm25`                              | `_nl_woRet.txt` |
| w/oMod  | `--no_sentences --no_dynamic_bm25 --no_few_shot` | `_nl_woMod.txt` |

The output filename automatically carries the variant tag (`variant_tag()`); to reproduce all
four cells of the RQ5 table, run the script four times with the four flag settings.

## Data files expected (per language)

`load_merged_data()` opens the following files under `--data_root` (default `./data/LowData`);
**only the ones strictly needed by the current toggle combination are read**. `<base>` is the
per-model folder tag (`qwen3th`, `ds_small3th`, …):

| File | Always | Needed when |
|---|:---:|---|
| `<data_root>/<lang>/trans_<base>/<lang>_python_best_candidate_vllm_B0.5_S0.5.jsonl` | ✅ | (always — translated pivot code) |
| `<data_root>/<lang>/trans_<base>/python_<lang>_structure_B0.5_S0.5_preds.jsonl` |   | `--use_sentences` |
| `<data_root>/<lang>/trans_<base>/python_<lang>_BM25_results_sentences_preds.jsonl` |   | `--use_few_shot AND --use_dynamic_bm25` |

If a required file is missing, the script logs an error and skips that language.

## Models & languages

`--models` defaults to all five LLMs under `./models/<name>` (or pass HuggingFace Hub names
like `Qwen/Qwen2.5-Coder-14B-Instruct`). `--langs` defaults to
`['julia','lua','ocaml','r','racket']`; pass `--langs ruby` (with `--data_root ./data/CSN`) for
the Ruby benchmark.

## Quick start (Full method)

```bash
# Defaults already match Full:
python finalScript_genNL.py
# → writes ./results/<model>/<lang>/<lang>_nl_full.txt   +   *_debug.jsonl

# A single ablation cell, one model, one language:
python finalScript_genNL.py --no_dynamic_bm25 \
    --models ./models/Qwen2.5-Coder-14B-Instruct --langs racket
```

## ICSE-2025 prompt adapter

`gen_NL_sentences_vllm_icse_prompt_adapter.py` runs the same Full pipeline but built around the
ICSE-2025 template (system + few-shot + task block) from Makharev & Ivanov 2025. It shares the
same `--models / --langs / --data_root / --output_root` CLI; output suffix is `_nl_icse2025_prompt.txt`.

## Output

For each (model × language), the script writes:
- `./results/<model_short_name>/<lang>/<lang>_nl_<variant>.txt` — one `idx\tsummary` per line
- `./results/<model_short_name>/<lang>/<lang>_nl_<variant>_debug.jsonl` — full prompt + raw output for inspection

These `.txt` files are the inputs to `06_evaluation/`.
