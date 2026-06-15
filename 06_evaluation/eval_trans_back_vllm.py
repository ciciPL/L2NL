import os
import json
import ast
import numpy as np
import pandas as pd

# ================= Defaults (overridable via CLI) =================
DEFAULT_MODELS = [
    "./models/deepseek-coder-1.3b-instruct",
    "./models/deepseek-coder-6.7b-instruct",
    "./models/Llama-3.1-8B-Instruct",
    "./models/Seed-Coder-8B-Instruct",
    "./models/Qwen2.5-Coder-14B-Instruct",
]
DEFAULT_LANGS       = ['julia', 'lua', 'ocaml', 'racket', 'r']
DEFAULT_DATA_ROOT   = "./data/LowData"
DEFAULT_BEST_SUFFIX = "_python_best_candidate_vllm_B0.1_S0.9.jsonl"
# ==================================================================


def _base_subdir(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))

def check_syntax(code):
    """AST 编译率检测"""
    if not code or not str(code).strip(): return 0
    try:
        ast.parse(code)
        return 1
    except:
        return 0


def get_scores_safe(candidate_dict):
    """从字典中安全读取已保存的分数"""
    if not candidate_dict: return 0.0, 0.0
    scores = candidate_dict.get('scores', {})
    if not scores: return 0.0, 0.0
    return scores.get('bleu', 0.0), scores.get('jina', 0.0)


def main(models, langs, data_root, best_suffix):
    for MODEL_PATH in models:
        model_base_name = os.path.basename(MODEL_PATH)
        base_path = _base_subdir(MODEL_PATH)
        all_stats = []

        for lang in langs:
            base_dir = f"{data_root}/{lang}/trans_{base_path}"
            target_file = f"{base_dir}/{lang}{best_suffix}"

            if not os.path.exists(target_file):
                print(f"Skipping {lang} (File not found)")
                print(target_file)
                continue

            stats = {
                "base_valid": [], "base_bleu": [], "base_jina": [],
                "ours_valid": [], "ours_bleu": [], "ours_jina": []
            }

            count = 0
            with open(target_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    count += 1

                    # --- 1. Base (t=0) ---
                    candidates = item.get('translations', [])
                    base_cand = None
                    for cand in candidates:
                        t = cand.get('temperature')
                        if t is not None and abs(t - 0.0) < 1e-6:
                            base_cand = cand
                            break
                    if not base_cand and candidates: base_cand = candidates[0]

                    base_code = base_cand.get('code', "") if base_cand else ""
                    b_bleu, b_jina = get_scores_safe(base_cand)

                    stats["base_valid"].append(check_syntax(base_code))
                    stats["base_bleu"].append(b_bleu)
                    stats["base_jina"].append(b_jina)

                    # --- 2. Ours (Best) ---
                    best_info = item.get('best_candidate_info', {})
                    our_code = best_info.get('python_code', "")
                    if not our_code: our_code = item.get('best_python_code', "")
                    o_bleu, o_jina = get_scores_safe(best_info)

                    stats["ours_valid"].append(check_syntax(our_code))
                    stats["ours_bleu"].append(o_bleu)
                    stats["ours_jina"].append(o_jina)

            # 计算平均值
            row = {
                "Lang": lang,
                "Base_CR": np.mean(stats["base_valid"]),
                "Base_BLEU": np.mean(stats["base_bleu"]),
                "Base_SBERT": np.mean(stats["base_jina"]),
                "Our_CR": np.mean(stats["ours_valid"]),
                "Our_BLEU": np.mean(stats["ours_bleu"]),
                "Our_SBERT": np.mean(stats["ours_jina"])
            }
            all_stats.append(row)

        if not all_stats:
            print("No data processed.")
            continue

        # ================= 输出表格 (完美复刻你的格式) =================
        df = pd.DataFrame(all_stats)
        avg_series = df.mean(numeric_only=True)
        avg_row = avg_series.to_dict()
        avg_row["Lang"] = "AVERAGE"
        display_rows = all_stats + [avg_row]

        print("\n" + "=" * 115)
        # 这里为了保持一致，标题也用你原来的风格
        print(f"Detailed Results ({base_path}) - Mode: {model_base_name}")
        print("=" * 115)

        # 你的经典表头：CR 宽10，BLEU/SBERT 宽12
        header = f"{'Language':<10} | {'Base CR':<10} {'Our CR':<10} | {'Base BLEU':<12} {'Our BLEU':<12} | {'Base SBERT':<12} {'Our SBERT':<12}"
        print(header)
        print("-" * 115)

        for row in display_rows:
            label = row['Lang']
            if label == "AVERAGE":
                print("-" * 115)

            print(f"{label:<10} | "
                  f"{row['Base_CR'] * 100:<10.2f} {row['Our_CR'] * 100:<10.2f} | "
                  f"{row['Base_BLEU']:<12.4f} {row['Our_BLEU']:<12.4f} | "
                  f"{row['Base_SBERT']:<12.4f} {row['Our_SBERT']:<12.4f}")

        print("=" * 115)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Base (τ=0) vs Best (multi-τ + back-translation selection) per (model, lang). "
                    "Reports AST compile-rate / BLEU / SBERT for §5.1 RQ1."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Translator model paths whose best-candidate JSONLs to read.")
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS,
                        help="Languages to process. Use ['ruby'] for CSN run.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                        help="Root holding <lang>/trans_<base>/... (default: ./data/LowData).")
    parser.add_argument("--best_suffix", default=DEFAULT_BEST_SUFFIX,
                        help="Best-candidate file suffix (default: %(default)s).")
    args = parser.parse_args()

    main(args.models, args.langs, args.data_root, args.best_suffix)