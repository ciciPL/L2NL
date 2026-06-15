# Baselines (Paper §4.4)

The paper's RQ4 compares the proposed method against three baseline groups (§4.4.1–§4.4.3).
**This directory contains no baseline source code**; reproduction instructions point to the
original authors' repositories or to the numbers reported in the literature.

## Group A — LLM prompting baselines (§4.4.1)

Two intrinsic variants of our own pipeline:

| Baseline | How to reproduce |
|---|---|
| **Zero-shot** | Run `05_summary_gen/finalScript_genNL.py` with `USE_FEW_SHOT=False`, `USE_SENTENCES=False`, `USE_DYNAMIC_BM25_SHOTS=False` (the `w/oMod` variant). |
| **Few-shot (static)** | Run `05_summary_gen/finalScript_genNL.py` with `USE_FEW_SHOT=True`, `USE_DYNAMIC_BM25_SHOTS=False`, `USE_SENTENCES=False` (no key-logic trace, k=3 fixed examples from `FEW_SHOTS_DATA`). |

## Group B — Open-source reproducible methods (§4.4.2)

Trained on PCSD, evaluated on the cleaned LRPL test sets produced by `01_data_prep/`.

### EACS — Sun et al., 2024 (ref19)

> Sun, W. et al. *An Extractive-and-Abstractive Framework for Source Code Summarization.*
> ACM TOSEM 33(3), 2024.

**Repository:** https://github.com/wssun/EACS

Steps:
1. Clone the EACS repo and install its dependencies.
2. Train on PCSD with the authors' default config.
3. Replace the test set with our cleaned LRPL set (`./data/LowData/...` produced by `01_data_prep/`).
4. Evaluate with our `06_evaluation/eval_lrcs_jar.py` (LRCS/RaxCS-compatible metrics).

### ESALE — Fang et al., 2024 (ref17)

> Fang, C. et al. *ESALE: Enhancing Code-Summary Alignment Learning for Source Code Summarization.*
> IEEE TSE 50(8), 2024.

**Repository:** https://github.com/NTDXYG/ESALE

Same procedure as EACS.

### EACS-TRANS / ESALE-TRANS (pivot variants)

For the pivot-translated variants in Table 7 (§4.5 RQ3):

```bash
# 1. Translate LRPL → Python pivot
python 02_translation/translate_stepBystep_vllm.py
python 02_translation/translate_find_bestCode_from_back_vllm.py
# 2. Feed the resulting *_python_best_candidate_*.jsonl to EACS / ESALE inference
#    (instead of the raw LRPL code), then evaluate as above.
```

## Group C — CSN Ruby published baselines (§4.4.3)

Per the paper:

> "This group of baselines is used for horizontal comparison on the public CSN benchmark.
> To ensure consistency of data splits and fairness of comparison, the results of these
> comparison methods are uniformly taken from the report of RaxCS."

**No re-implementation is required.** The numbers cited in Table 7 are taken verbatim from:

| Baseline | Source of cited numbers |
|---|---|
| Seq2Seq (ref58) | RaxCS paper Table |
| IR-direct (ref59, ref60) | RaxCS paper Table |
| Re2Com (ref61) | RaxCS paper Table |
| RoBERTa (ref62) | RaxCS paper Table |
| CodeBERT (ref04) | RaxCS paper Table |
| CodeT5 (ref06) | RaxCS paper Table |
| RaxCS (ref23) | Yang et al., *Information and Software Technology*, 2025 |
| LRCS (ref10) | Guo et al., *Software: Practice and Experience*, 2024 — values from the original LRCS paper |

Our method's CSN Ruby numbers in Table 7 are computed with `06_evaluation/eval_lrcs_jar.py`
(the LRCS-compatible local-implementation evaluator), so they are directly comparable to the
above reported values.
