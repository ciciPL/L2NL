# Evaluation (Paper §4.2)

Two summary evaluators (used for different experiments) + AST parsability + back-translation
analysis (Fig 5) + self-contained metric libraries.

## Summary evaluators

| Script | Used for | Why this implementation |
|---|---|---|
| **`eval_hf_real_ref.py`** | **§4.5 RQ2 — LRPL prompting comparison.** Compares our prompting variants (Full / w/oKey / w/oRet / w/oMod) against each other across the five LRPL test sets. | HuggingFace `evaluate` library (unified BLEU/ROUGE/METEOR/BERTScore). All variants share one metric implementation → clean within-paper comparison. |
| **`eval_lrcs_jar.py`** | **§4.5 RQ4 — Ruby benchmark vs published baselines.** Compares our method against LRCS / RaxCS / CodeT5 / Re2Com / IR-direct / Seq2Seq / RoBERTa / CodeBERT on CSN Ruby. | Local libraries matching the LRCS / RaxCS papers: CodeXGLUE BLEU (`bleu.py`) + METEOR-1.5 Java jar (`meteor/`) + local ROUGE-L (`rouge/`). Direct apples-to-apples with baselines' *reported* numbers. |

The two scripts produce slightly different absolute numbers due to the different underlying
metric implementations. This is intentional — each is valid within its own use case and should
not be cross-used.

> **Local suite metrics:** `eval_lrcs_jar.py` computes BLEU (CodeXGLUE `bleu.py::bleuFromMaps`),
> METEOR (Java jar) and ROUGE-L (local `rouge.py`). BERTScore for the local suite is provided
> separately by `myEvaluate/MyscoreBert.py`.

## Structural & translation-side metrics

| Script | Used for | Output |
|---|---|---|
| **`ast_acc_deepseek.py`** | **§4.5 RQ1 / §4.2 — APR** (AST Parsability Rate). Parses the greedy (τ=0) pivot candidate with `ast.parse`; sample is parsable iff it parses without error. Denominator is the actual evaluated count, or a fixed `--sample_count`. | per-model × per-language APR table |
| **`eval_trans_back_vllm.py`** | **§5.1 RQ1 — Base (τ=0) vs Best (multi-τ + back-translation selection).** For each (model × language), reads the best-candidate JSONL from `02_translation/`, reports AST compile-rate / back-translation BLEU / SBERT for the τ=0 baseline candidate vs the selected best candidate. | Per-model console table |

## Bundled metric libraries

| Path | Provides | Used by |
|---|---|---|
| `bleu.py` | CodeXGLUE/MOSES BLEU (`computeMaps` / `bleuFromMaps` / `splitPuncts`) | `eval_lrcs_jar.py` (`import bleu`) |
| `meteor/` | METEOR (`meteor-1.5.jar` + paraphrase data + `meteor.py::Meteor`) | `eval_lrcs_jar.py` via Java subprocess |
| `rouge/` | Local ROUGE-L wrapper (`rouge.py::Rouge`) | `eval_lrcs_jar.py` (`from rouge.rouge import Rouge`) |
| `myEvaluate/MyscoreBert.py` | Functional BERTScore wrapper (DeBERTa-xlarge-mnli) | local-suite BERTScore (run separately) |
| `myEvaluate/scorer.py` | BERTScore `BERTScorer` class (third-party `bert_score` source) | kept for completeness |
| `myEvaluate/utils.py` | Low-level BERTScore helpers (`bert_cos_score_idf`, `get_bert_embedding`, …) | `MyscoreBert.py`, `scorer.py` |

## Quick start

```bash
# (a) §4.5 RQ2: LRPL prompting variants comparison (HF evaluate)
python eval_hf_real_ref.py
# predictions: ./results/<model>/<lang>/<lang>_nl_<variant>.txt   (see --targets)
# references : ./data/LowData/step3_final_result/<lang>_final.tsv

# (b) §4.5 RQ4: Ruby benchmark vs LRCS / RaxCS / CodeT5 (local jar suite)
python eval_lrcs_jar.py --langs ruby \
    --ref_template "./data/CSN/{lang}/trans_qwen3th/test.gold" \
    --hyp_template "./results/Qwen2.5-Coder-14B-Instruct/{lang}/{lang}_nl_full.txt"
# prints METEOR + ROUGE-L (BLEU once bleu.py is added)

# (c) §4.5 RQ1: AST Parsability Rate
python ast_acc_deepseek.py
# scans ./data/LowData/<lang>/trans_<base>/

# (d) §5.1 RQ1: Base (τ=0) vs Best (multi-τ + back-translation) summary table
python eval_trans_back_vllm.py
# reads ./data/LowData/<lang>/trans_<base>/<lang>_python_best_candidate_vllm_B0.1_S0.9.jsonl
# prints per-model AST CR / BLEU / SBERT comparison to stdout
# (Fig 5 itself — per-τ Mean vs Oracle trajectory — is plotted by
#  10_figures/plot_temperature_picture.py, which reads the same JSONL directly.)
```

## Paths

All scripts accept (or default to) paths under `./data/`, `./models/`, `./results/`. Pass paths
via argparse (every script now has a CLI; `--help` lists the flags).

For BERTScore, the HF evaluator defaults to the HuggingFace Hub name `microsoft/deberta-xlarge-mnli`;
a local copy at `./models/microsoft/deberta-xlarge-mnli` is honoured if present.
