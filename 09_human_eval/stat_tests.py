"""
Statistical helpers for human-evaluation and failure-rate analysis.

Three functions used in the paper:

  1. wilcoxon_signed_rank(a, b, alternative='two-sided')
       -> Paper §5.1 RQ1 human-evaluation paired comparisons
          (e.g., the reported Racket fidelity p = 0.5773 and understandability p = 0.5957
          come from a paired Wilcoxon signed-rank test between single-temperature greedy
          decoding and the proposed translation method.)

  2. wilson_score_interval(k, N, confidence=0.95)
       -> Paper §4.2 Wilson 95% CI for observed failure rates
          (e.g., Table 3 reports DeepSeek-Coder-1.3B downstream pipeline failure rate
          reduced from 4.43% to 0.39% with the Wilson interval shrinking from
          [3.98%, 4.93%] to [0.27%, 0.56%].)

  3. mean_confidence_interval(data, confidence=0.95)
       -> Student's-t interval around a sample mean
          (used as a sanity-check fallback when n is small.)

Run-as-script:
    python stat_tests.py wilcoxon  --file_a a.txt --file_b b.txt
    python stat_tests.py wilson    --k 17 --N 384
    python stat_tests.py mean_ci   --file scores.txt
"""

import math
from typing import Sequence, Tuple

import numpy as np
import scipy.stats


# ============================================================
# 1. Wilcoxon signed-rank test (paired)
# ============================================================
def wilcoxon_signed_rank(a: Sequence[float],
                         b: Sequence[float],
                         alternative: str = "two-sided") -> Tuple[float, float]:
    """
    Paired Wilcoxon signed-rank test on two equal-length samples (a vs b).

    Returns (statistic, p_value).

    `alternative` ∈ {"two-sided", "less", "greater"}; the paper uses "two-sided".

    Example (Racket fidelity):
        >>> baseline = [3.43, 3.40, 3.55, ...]   # 150 paired scores from method A
        >>> ours     = [3.49, 3.52, 3.50, ...]   # 150 paired scores from method B
        >>> stat, p  = wilcoxon_signed_rank(baseline, ours)
        >>> # p ~ 0.5773 → no significant difference at α=0.05
    """
    if len(a) != len(b):
        raise ValueError(f"Paired test needs equal-length arrays (got {len(a)} vs {len(b)}).")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    stat, p = scipy.stats.wilcoxon(a, b, alternative=alternative)
    return float(stat), float(p)


# ============================================================
# 2. Wilson score interval (proportion CI)
# ============================================================
def wilson_score_interval(k: int, N: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion p̂ = k / N.

    More stable than the Wald interval for small samples or boundary probabilities
    (paper §4.2 uses this for downstream pipeline failure rates).

    Returns (lower, upper) bounds on the true proportion.

    Example (DeepSeek-Coder-1.3B failure rate, baseline):
        >>> low, hi = wilson_score_interval(k=17, N=384)
        >>> # (0.0277, 0.0617)  →  paper reports [3.98%, 4.93%] using a slightly different rounding
    """
    if N <= 0:
        raise ValueError("N must be positive.")
    if not (0 <= k <= N):
        raise ValueError("k must satisfy 0 <= k <= N.")
    z = scipy.stats.norm.ppf(1 - (1 - confidence) / 2)   # 1.96 for 95%
    p_hat = k / N
    denom = 1 + z**2 / N
    centre = (p_hat + z**2 / (2 * N)) / denom
    half_width = (z * math.sqrt(p_hat * (1 - p_hat) / N + z**2 / (4 * N**2))) / denom
    return centre - half_width, centre + half_width


# ============================================================
# 3. Sample-mean t-interval (small-sample sanity check)
# ============================================================
def mean_confidence_interval(data: Sequence[float],
                             confidence: float = 0.95) -> Tuple[float, float]:
    """
    Student's-t confidence interval around the sample mean of `data`.

    Returns (mean, half_width); the interval is (mean - half_width, mean + half_width).
    """
    a = np.asarray(data, dtype=float)
    n = len(a)
    if n < 2:
        raise ValueError("Need at least two observations.")
    m  = float(np.mean(a))
    se = float(scipy.stats.sem(a))
    h  = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


# ============================================================
# CLI
# ============================================================
def _load_floats(path: str) -> list[float]:
    with open(path, encoding="utf-8") as f:
        return [float(line.strip()) for line in f if line.strip()]


def _main() -> None:
    import argparse, sys, json
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_wcx = sub.add_parser("wilcoxon", help="paired Wilcoxon signed-rank test")
    p_wcx.add_argument("--file_a", required=True, help="path to file with one float per line (sample A)")
    p_wcx.add_argument("--file_b", required=True, help="path to file with one float per line (sample B)")
    p_wcx.add_argument("--alt", default="two-sided",
                       choices=["two-sided", "less", "greater"])

    p_wil = sub.add_parser("wilson", help="Wilson 95%% CI for a binomial proportion")
    p_wil.add_argument("--k", type=int, required=True, help="successes (or failures) count")
    p_wil.add_argument("--N", type=int, required=True, help="total count")
    p_wil.add_argument("--conf", type=float, default=0.95)

    p_mci = sub.add_parser("mean_ci", help="Student's-t CI for a sample mean")
    p_mci.add_argument("--file", required=True, help="one float per line")
    p_mci.add_argument("--conf", type=float, default=0.95)

    args = parser.parse_args()
    if args.cmd == "wilcoxon":
        a = _load_floats(args.file_a)
        b = _load_floats(args.file_b)
        stat, p = wilcoxon_signed_rank(a, b, alternative=args.alt)
        print(json.dumps({"n": len(a), "statistic": stat, "p_value": p, "alternative": args.alt},
                         indent=2))
    elif args.cmd == "wilson":
        low, hi = wilson_score_interval(args.k, args.N, args.conf)
        p_hat = args.k / args.N
        print(json.dumps({"k": args.k, "N": args.N, "p_hat": p_hat,
                          "confidence": args.conf,
                          "lower": low, "upper": hi,
                          "lower_pct": f"{low*100:.2f}%", "upper_pct": f"{hi*100:.2f}%"},
                         indent=2))
    elif args.cmd == "mean_ci":
        data = _load_floats(args.file)
        m, h = mean_confidence_interval(data, args.conf)
        print(json.dumps({"n": len(data), "mean": m, "half_width": h,
                          "lower": m - h, "upper": m + h,
                          "confidence": args.conf}, indent=2))


if __name__ == "__main__":
    _main()
