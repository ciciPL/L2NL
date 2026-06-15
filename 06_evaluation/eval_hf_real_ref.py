# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import numpy as np
import torch
import evaluate

# =========================
# 全局配置
# =========================
RANDOM_SEED = 42
SAMPLE_SIZE = 9999


# =========================
# 工具函数 (保持不变)
# =========================

def load_metrics():
    print(">>> Initializing HuggingFace `evaluate` metrics...")
    try:
        return (
            evaluate.load("bleu"),
            evaluate.load("rouge"),
            evaluate.load("meteor"),
            evaluate.load("bertscore"),
        )
    except Exception as e:
        print(f"Metric loading failed: {e}")
        return None


def compute_scores(metric_bundle, refs, preds, model_path):
    m_bleu, m_rouge, m_meteor, m_bert = metric_bundle
    results = {}
    if not preds or not refs:
        return {k: 0.0 for k in ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore']}

    try:
        results['BLEU'] = m_bleu.compute(predictions=preds, references=[[r] for r in refs])['bleu']
    except:
        results['BLEU'] = 0.0

    try:
        results['ROUGE-L'] = m_rouge.compute(predictions=preds, references=refs)['rougeL']
    except:
        results['ROUGE-L'] = 0.0

    try:
        results['METEOR'] = m_meteor.compute(predictions=preds, references=refs)['meteor']
    except:
        results['METEOR'] = 0.0

    try:
        # results['BERTScore'] = 0.0
        # 如果需要启用 BERTScore，取消下面注释
        bert_res = m_bert.compute(
            predictions=preds, references=refs, model_type=model_path, num_layers=40,
            lang="en", batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        results['BERTScore'] = float(np.mean(bert_res['f1']))
    except:
        results['BERTScore'] = 0.0

    return results


def read_hyp_lines(path):
    if not os.path.exists(path):
        return []
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                lines.append("")
                continue
            # 按 index \t summary 切分
            parts = line.split('\t', 1)
            if len(parts) == 2:
                _, summary = parts
            else:
                # 如果没有 tab，整行当 summary（防止异常文件）
                summary = parts[0]
            summary = summary.replace("Summary", "").strip()
            lines.append(summary)
    return lines


# =========================
# 诊断模式
# =========================
def debug_check(df_ref, hyp_lines, model_name, method_name):
    print(f"\n🚨 [诊断模式] {model_name} - {method_name} 对齐检查")

    if 3 in df_ref['index'].values:
        test_id = 3
    else:
        test_id = df_ref['index'].iloc[0]

    ref_row = df_ref[df_ref['index'] == test_id].iloc[0]
    ref_txt = ref_row['summary']

    target_idx = test_id

    print(f"   测试 ID: {test_id}")
    print(f"   REF (真值): {ref_txt[:60]}...")

    if target_idx < len(hyp_lines):
        hyp_txt = hyp_lines[target_idx]
        print(f"   HYP (第{target_idx + 1}行): {hyp_txt[:60]}...")
    else:
        print(f"   HYP: [Index超出范围 len={len(hyp_lines)}]")

    print("   ------------------------------------------------")


# =========================
# 主流程
# =========================

def _parse_targets(target_args):
    """Parse 'DisplayName:template_with_{lang}' strings into (name, template) tuples."""
    targets = []
    for t in target_args:
        if ":" not in t:
            raise ValueError(f"--targets entry must be 'Name:template', got: {t}")
        name, template = t.split(":", 1)
        targets.append((name.strip(), template.strip()))
    return targets


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Paper §4.5 RQ2 — within-paper LRPL prompting-variant comparison using the "
                    "HuggingFace `evaluate` library (BLEU/ROUGE-L/METEOR/BERTScore)."
    )
    parser.add_argument("--models", nargs="+",
                        default=["deepseek-coder-1.3b-instruct", "deepseek-coder-6.7b-instruct",
                                 "Llama-3.1-8B-Instruct", "Seed-Coder-8B-Instruct",
                                 "Qwen2.5-Coder-14B-Instruct"],
                        help="Model short-names (sub-dirs under --results_root).")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'r', 'racket'])
    parser.add_argument("--results_root", default="./results",
                        help="Predictions live at <results_root>/<model>/<lang>/<file>.")
    parser.add_argument("--ref_root", default="./data/LowData/step3_final_result",
                        help="References live at <ref_root>/<lang>_final.tsv.")
    parser.add_argument("--bertscore_model", default="microsoft/deberta-xlarge-mnli",
                        help="model_type passed to BERTScore.")
    parser.add_argument("--sample_size", type=int, default=9999,
                        help="Cap on references per language (default: effectively all).")
    parser.add_argument("--targets", nargs="+",
                        default=[
                            "Full:{lang}_nl_full.txt",
                            "w/oKey:{lang}_nl_woKey.txt",
                            "w/oRet:{lang}_nl_woRet.txt",
                            "w/oMod:{lang}_nl_woMod.txt",
                        ],
                        help="Comparison columns as 'DisplayName:filename_template' entries; "
                             "{lang} is substituted. Filenames are the outputs of "
                             "05_summary_gen/finalScript_genNL.py.")
    args = parser.parse_args()

    global SAMPLE_SIZE
    SAMPLE_SIZE = args.sample_size
    BASE_MODEL_PATH = args.bertscore_model
    MODELS = args.models
    LANGS  = args.langs
    EVAL_TARGETS = _parse_targets(args.targets)

    metrics = load_metrics()
    if not metrics: return

    for model_name in MODELS:
        print(f"\n{'=' * 30} 模型: {model_name} {'=' * 30}")

        for lang in LANGS:
            print(f"\n>>> 语言: {lang.upper()}")

            # 1. Load references
            ref_path = f"{args.ref_root}/{lang}_final.tsv"
            if not os.path.exists(ref_path):
                print(f"[skip] missing REF: {ref_path}")
                continue

            df_ref = pd.read_csv(ref_path, sep='\t', on_bad_lines='skip')
            if len(df_ref) > SAMPLE_SIZE:
                df_ref = df_ref.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

            # 2. 动态读取 HYP
            hyp_raw_data = {}
            # 记录第一个读取成功的 Key，用于诊断
            first_valid_key = None

            for display_name, file_template in EVAL_TARGETS:
                # Build prediction path
                filename = file_template.format(lang=lang)
                full_path = f"{args.results_root}/{model_name}/{lang}/{filename}"
                lines = read_hyp_lines(full_path)
                if lines:
                    hyp_raw_data[display_name] = lines
                    if first_valid_key is None:
                        first_valid_key = display_name
                else:
                    # 如果文件不存在，可以选择报错或者仅打印警告
                    # print(f"   [警告] 文件未找到: {display_name} -> {full_path}")
                    pass

            # 如果一个文件都没读到，跳过
            if not hyp_raw_data:
                print("⚠️ 未找到任何预测文件，跳过该语言。")
                continue

            # 对第一个有效数据做诊断
            if first_valid_key:
                debug_check(df_ref, hyp_raw_data[first_valid_key], f"{model_name}-{lang}", first_valid_key)

            # 3. 对齐逻辑 (求交集)
            # 收集每个模型有效的 ID (行号 >= ID)
            valid_ids_sets = []
            ref_ids_set = set(df_ref['index'].astype(int))

            for name, lines in hyp_raw_data.items():
                max_idx = len(lines) - 1
                valid_for_model = {uid for uid in ref_ids_set if uid <= max_idx}
                valid_ids_sets.append(valid_for_model)

            if not valid_ids_sets:
                print("⚠️ 无有效预测数据 ID")
                continue

            common_ids = set.intersection(*valid_ids_sets)

            if not common_ids:
                print("⚠️ ID 交集为空 (可能是某个文件行数太少，导致无法对齐)")
                continue

            sorted_ids = sorted(list(common_ids))
            print(f"   最终共同评测样本数: {len(sorted_ids)}")

            # 构建对齐后的数据列表
            ref_lookup = dict(zip(df_ref['index'], df_ref['summary']))
            final_refs = [ref_lookup[uid] for uid in sorted_ids]

            final_preds_dict = {name: [] for name in hyp_raw_data}
            for uid in sorted_ids:
                for name in hyp_raw_data:
                    final_preds_dict[name].append(hyp_raw_data[name][uid])

            # 4. 计算分数
            scores = {}
            # 按照 EVAL_TARGETS 的顺序计算，保证输出顺序一致
            for display_name, _ in EVAL_TARGETS:
                if display_name in final_preds_dict:
                    scores[display_name] = compute_scores(metrics, final_refs, final_preds_dict[display_name],
                                                          BASE_MODEL_PATH)

            # 5. 动态输出表格
            # ----------------------------------------------------
            # 动态计算表格宽度和格式
            # ----------------------------------------------------
            col_width = 12
            # 只有那些成功加载并计算出分数的列才显示
            valid_columns = [t[0] for t in EVAL_TARGETS if t[0] in scores]

            # 也就是: Metric列 + 竖线 + (数据列 * N)
            total_width = col_width + 3 + (len(valid_columns) * (col_width + 3))

            print("\n" + "-" * total_width)
            print(f"{lang.upper()} Result Comparison (Model: {model_name})")
            print("-" * total_width)

            # 构建表头
            header_str = f"{'Metric':<{col_width}}"
            for col_name in valid_columns:
                header_str += f" | {col_name:>{col_width}}"
            print(header_str)
            print("-" * total_width)

            # 构建数据行
            for metric in ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore']:
                row_str = f"{metric:<{col_width}}"
                for col_name in valid_columns:
                    val = scores[col_name].get(metric, 0.0)
                    row_str += f" | {val * 100:>{col_width}.4f}"
                print(row_str)
            print("-" * total_width)


if __name__ == "__main__":
    main()
