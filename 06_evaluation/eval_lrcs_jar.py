"""
Paper §4.5 RQ4 — LRCS/RaxCS-compatible local evaluator.

This is the local-library evaluation suite used for the Ruby (CSN) benchmark and the
LRPL sets, kept separate from the HuggingFace `evaluate` suite (`eval_hf_real_ref.py`)
so our numbers are directly comparable to the values reported by LRCS / RaxCS.

Metrics:
  - BLEU    : CodeXGLUE/MOSES `bleu.py::bleuFromMaps` (same implementation as the cited
              baselines; smooth=1).
  - METEOR  : METEOR-1.5 reference Java implementation, via `meteor/meteor-1.5.jar`
              (wrapped by `meteor/meteor.py::Meteor`).
  - ROUGE-L : local `rouge/rouge.py::Rouge` (same implementation as the cited baselines).
  - BERTScore: see `myEvaluate/MyscoreBert.py` (run separately).

Input format (one summary per line, line index = sample id):
  ref : `<ref_template>`  e.g. dataset/CSN/<lang>/.../test.gold
  hyp : `<hyp_template>`  e.g. the *.txt produced by 05_summary_gen/

Usage:
    python eval_lrcs_jar.py --langs ruby \\
        --ref_template "./data/CSN/{lang}/trans_qwen3th/test.gold" \\
        --hyp_template "./results/Qwen2.5-Coder-14B-Instruct/{lang}/{lang}_nl_full.txt"
"""

import os
import sys

# Make sibling packages (meteor/, rouge/) importable when run from this directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from meteor.meteor import Meteor
from rouge.rouge import Rouge
import bleu
from bleu import splitPuncts


def load_pairs(ref_path, hyp_path):
    """Read aligned ref/hyp files into {id: [sentence]} dicts (METEOR/ROUGE API format)."""
    gts, res = {}, {}
    cnt = 0
    print(f"Reading data from:\n  ref: {ref_path}\n  hyp: {hyp_path}")
    with open(hyp_path, "r", encoding="utf-8") as hyp_file, \
         open(ref_path, "r", encoding="utf-8") as ref_file:
        for hyp, ref in zip(hyp_file, ref_file):
            if hyp == '\n':
                continue
            res[cnt] = [hyp.strip()]
            gts[cnt] = [ref.strip()]
            cnt += 1
    print(f"Valid samples: {cnt}")
    return gts, res


def evaluate_lang(lang, ref_path, hyp_path):
    if not os.path.exists(ref_path) or not os.path.exists(hyp_path):
        print(f"[skip] {lang}: missing ref or hyp\n  ref: {ref_path}\n  hyp: {hyp_path}")
        return
    gts, res = load_pairs(ref_path, hyp_path)
    if not gts:
        print(f"[skip] {lang}: no valid pairs")
        return

    print(f"\n========== {lang.upper()} ==========")

    # ---- BLEU (CodeXGLUE bleu.py) ----
    # bleuFromMaps expects each value to be a list of cooked references, i.e.
    # {id: [splitPuncts(lowercased_text)]} — the exact format computeMaps() builds.
    try:
        predictionMap = {i: [splitPuncts(res[i][0].strip().lower())] for i in res}
        refMap        = {i: [splitPuncts(gts[i][0].strip().lower())] for i in gts}
        bleu_score = bleu.bleuFromMaps(refMap, predictionMap)
        print("BLEU    =", round(bleu_score[0], 2))
    except Exception as e:
        print(f"Error calling BLEU: {e}")

    # ---- METEOR (Java jar) ----
    try:
        meteor_score, _ = Meteor().compute_score(gts, res)
        print("METEOR  =", round(meteor_score * 100, 2))
    except Exception as e:
        print(f"Error calling METEOR: {e}")

    # ---- ROUGE-L (local rouge.py) ----
    try:
        rouge_score, _ = Rouge().compute_score(gts, res)
        print("ROUGE-L =", round(rouge_score * 100, 2))
    except Exception as e:
        print(f"Error calling ROUGE: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="LRCS/RaxCS-compatible local evaluator (METEOR jar + ROUGE-L; BLEU via bleu.py)."
    )
    parser.add_argument("--langs", nargs="+", default=['ruby'],
                        help="Languages to evaluate (default: ruby for the CSN benchmark).")
    parser.add_argument("--ref_template", required=True,
                        help="Reference file template with {lang}, one summary per line.")
    parser.add_argument("--hyp_template", required=True,
                        help="Hypothesis file template with {lang}, one summary per line.")
    args = parser.parse_args()

    for lang in args.langs:
        evaluate_lang(lang,
                      args.ref_template.format(lang=lang),
                      args.hyp_template.format(lang=lang))


if __name__ == "__main__":
    main()
