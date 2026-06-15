"""
Paper §4.5 RQ1 / §4.2 — AST Parsability Rate (APR).

For each (model × language), take the greedy (τ=0) Python pivot candidate from the
back-translation JSONL and check whether it parses with `ast.parse`. APR is the
fraction of parsable samples.

The denominator can be either the actual number of evaluated samples (default) or a
fixed per-language sample count passed via --sample_count (use this to reproduce the
paper's table exactly if it normalised over a fixed test-set size).
"""

import os
import json
import ast
import pandas as pd
from tqdm import tqdm


def _base_subdir(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))

def evaluate_model(model_path, langs, data_root, sample_count=None):
    base = _base_subdir(model_path)
    print(f"\nAPR evaluation for: {model_path}  (folder tag: trans_{base})")

    all_stats = []
    for lang in langs:
        base_dir = f"{data_root}/{lang}/trans_{base}"
        back_file = f"{base_dir}/{lang}_python_multi_trans_vllm_back.jsonl"

        if not os.path.exists(back_file):
            print(f"  [skip] {back_file} not found")
            continue

        total_count = 0
        valid_count = 0

        with open(back_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc=f"Processing {lang}", unit="item"):
            if not line.strip():
                continue
            item = json.loads(line)

            # Pick the greedy (τ=0) candidate
            translations = item.get('translations', [])
            target_cand = None
            for cand in translations:
                if cand.get('temperature', -1) in (0, 0.0):
                    target_cand = cand
                    break
            if not target_cand and translations:
                first = translations[0]
                if first.get('temperature') in (0, 0.0):
                    target_cand = first

            if target_cand:
                total_count += 1
                source_text = target_cand.get('code', '')
                try:
                    ast.parse(source_text)
                    valid_count += 1
                except SyntaxError:
                    pass

        # Denominator: fixed --sample_count if given, else the actual evaluated count.
        denom = sample_count if sample_count else total_count
        pass_rate = (valid_count / denom * 100) if denom > 0 else 0.0

        all_stats.append({
            "Language": lang,
            "Total": total_count,
            "Valid": valid_count,
            "Denominator": denom,
            "APR (%)": round(pass_rate, 2),
        })

    df = pd.DataFrame(all_stats)
    print("\n" + "=" * 60)
    print(f"APR result for {os.path.basename(model_path)}")
    print(df.to_string(index=False))
    print("=" * 60)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute AST Parsability Rate (APR) of the greedy (τ=0) Python pivot "
                    "candidate, per (model × language). Paper §4.5 RQ1 / §4.2."
    )
    parser.add_argument("--models", nargs="+",
                        default=[
                            "./models/deepseek-coder-1.3b-instruct",
                            "./models/deepseek-coder-6.7b-instruct",
                            "./models/Llama-3.1-8B-Instruct",
                            "./models/Seed-Coder-8B-Instruct",
                            "./models/Qwen2.5-Coder-14B-Instruct",
                        ])
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'racket', 'r'])
    parser.add_argument("--data_root", default="./data/LowData")
    parser.add_argument("--sample_count", type=int, default=None,
                        help="Fixed denominator for APR (e.g., the per-language test-set size). "
                             "If omitted, APR is normalised over the actual number of evaluated samples.")
    args = parser.parse_args()

    for model_path in args.models:
        evaluate_model(model_path, args.langs, args.data_root, args.sample_count)


if __name__ == "__main__":
    main()