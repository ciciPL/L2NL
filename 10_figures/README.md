# Figures (Paper §5.1, Fig 5)

Standalone matplotlib reproducer for **Fig 5 — Temperature × back-translation consistency**.

The paper's final figure is rendered in TikZ / pgfplots inside the manuscript,
but it consumes the same underlying numbers this script produces, so running
`plot_temperature_picture.py` is the simplest way to verify Fig 5 end-to-end.

## What it shows

For each (model × language) and each temperature τ ∈ {0, 0.7, 0.9, 1.1}, the
script reads the per-candidate back-translation scores stored in
`02_translation/`'s best-candidate JSONL and plots two panels:

| Panel | y-axis | Solid line ("Average") | Dashed line ("Oracle") |
|---|---|---|---|
| Left  | BLEU  | mean BLEU at τ  across candidates | max BLEU at τ  (theoretical upper bound) |
| Right | SBERT | mean SBERT at τ across candidates | max SBERT at τ (theoretical upper bound) |

The paper uses this evidence in §5.1 RQ1 to argue that single-temperature
greedy decoding (τ = 0) is sub-optimal: oracle curves stay flat or rise as τ
grows even though average curves drop, motivating multi-temperature sampling
with back-translation selection.

## Inputs

Default path layout (override with `--root` / `--suffix`):

```
./data/LowData/<lang>/trans_<base>/<lang>_python_best_candidate_vllm_B0.5_S0.5.jsonl
```

`<base>` is one of `ds_small3th`, `ds_mid3th`, `Llama3th`, `Seed3th`, `qwen3th`
(server-side experimental folder names — see `_base_subdir()` in the script).

These JSONL files are produced by:

```bash
python 02_translation/translate_stepBystep_vllm.py         # multi-τ sampling
python 02_translation/translate_back_2_lrpl_repair.py      # AST validation
python 02_translation/translate_find_bestCode_from_back_vllm.py  # BLEU+SBERT pick
```

## Usage

```bash
# Default (looks under ./data/LowData/, writes fig5_temperature_backtranslation.png)
python plot_temperature_picture.py

# Custom root / file suffix / output path
python plot_temperature_picture.py \
    --root ./data/LowData \
    --suffix _python_best_candidate_vllm_B0.5_S0.5.jsonl \
    --out fig5.png
```

Add `--show` to pop a window after saving.

## Dependencies

```
matplotlib, seaborn, pandas
```
