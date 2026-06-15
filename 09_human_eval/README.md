# Human Evaluation — Statistical Tests (Paper §4.2 + §5.1)

The actual human-evaluation pipeline (URL generator / batch scoring UI / LLM-as-judge data prep)
was a one-off tool chain and is **not** included here. What is preserved is the statistical
machinery used to report the paper's numbers.

## Single file: `stat_tests.py`

| Function | Used for | Paper anchor |
|---|---|---|
| `wilcoxon_signed_rank(a, b)` | Paired comparison of two methods' human-evaluation scores | §5.1 RQ1 (e.g., Racket fidelity p = 0.5773, understandability p = 0.5957) |
| `wilson_score_interval(k, N)` | Confidence interval for a binomial proportion (failure rate) | §4.2 Wilson 95% CI (e.g., Table 3 DeepSeek-Coder-1.3B failure rate `[3.98 %, 4.93 %]` → `[0.27 %, 0.56 %]`) |
| `mean_confidence_interval(data)` | Student's-t CI around a sample mean | Sanity-check fallback for small-n |

## Usage

```bash
# 1. Paired Wilcoxon (e.g., reproducing the Racket fidelity p-value)
python stat_tests.py wilcoxon --file_a baseline_fidelity.txt --file_b ours_fidelity.txt

# 2. Wilson 95% CI for a failure-rate proportion
python stat_tests.py wilson --k 17 --N 384
# → {"p_hat": 0.0443, "lower": 0.0277, "upper": 0.0617, ...}

# 3. Student's-t mean CI
python stat_tests.py mean_ci --file fidelity_scores.txt
```

Each input file is plain text with one float per line.

## Dependencies

```
numpy, scipy
```
