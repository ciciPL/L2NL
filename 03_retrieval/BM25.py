import argparse
import json
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# 1. 辅助函数：将 token 列表拼接成字符串 (用于生成 summary)
def makestr(lst):
    if not lst:
        return ""
    p = ""
    for w in lst:
        p = p + w + " "
    p = p.replace(" .", ".")
    return p.strip()


# 2. 核心函数：清洗 Python 代码 (严格防止 Docstring 泄露)
def clean_python_code(code, docstring=""):
    if not code:
        return ""

    # (1) 去除 Markdown 标记
    # code = re.sub(r'^```\w*\n', '', code)
    # code = re.sub(r'\n```$', '', code)

    # (2) 移除 """...""" 类型 docstring (处理多行、单行、缩进)
    # flags=re.MULTILINE 确保 ^ 能匹配每一行的开头
    code = re.sub(r'^\s*"""[\s\S]*?"""', '', code, flags=re.MULTILINE)

    # (3) 移除 '''...''' 类型 docstring
    code = re.sub(r"^\s*'''[\s\S]*?'''", '', code, flags=re.MULTILINE)

    # (4) 如果提供了原始 docstring 文本，尝试直接替换移除 (作为兜底)
    if docstring:
        code = code.replace(docstring, "")

    return code.strip()


def load_data(file_path):
    data = []
    print(f"Reading from {file_path}")
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_retrieval(args):
    # --- 1. 读取数据 ---
    # Train: 检索库 (Corpus) - CSN 原始数据
    print(f"Loading Corpus (Train) from: {args.train_file}")
    train_data = load_data(args.train_file)

    # Test: 查询集 (Query) - 你的翻译数据
    print(f"Loading Queries (Test) from: {args.test_file}")
    test_data = load_data(args.test_file)

    print(f"Train size: {len(train_data)}")
    print(f"Test size: {len(test_data)}")

    # --- 2. 预处理检索库 (Crucial Step: 提前清洗并保存) ---
    print("Preprocessing training corpus (Cleaning code)...")

    # 我们创建一个列表专门存储清洗后的代码，既用于 BM25，也用于最后结果输出
    # 这样可以保证存下来的 candidates 也是干净的
    train_corpus_cleaned = []

    for item in tqdm(train_data, desc="Cleaning Corpus"):
        raw_code = item.get('code', '')
        raw_doc = item.get('docstring', '')

        # 清洗代码
        cleaned = clean_python_code(raw_code, raw_doc)
        train_corpus_cleaned.append(cleaned)

    # --- 3. 构建 BM25 索引 ---
    print("Building BM25 index...")
    # 简单的空格分词，对于代码搜索通常够用，如果需要更精细可以使用 tree-sitter 或 nltk
    tokenized_corpus = [doc.split() for doc in train_corpus_cleaned]
    bm25 = BM25Okapi(tokenized_corpus)

    # --- 4. 执行检索并保存 ---
    output_file = args.output_file
    print(f"Retrieving top {args.top_k} and saving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, test_item in tqdm(enumerate(test_data), total=len(test_data), desc="Retrieving"):

            # A. 获取 Query 代码 (优先使用翻译好的 best_python_code)
            raw_query = test_item.get('best_python_code')
            if not raw_query:
                # 兜底：如果没有 best_python_code，尝试 source_code 或 code
                raw_query = test_item.get('source_code', test_item.get('code', ''))

            # B. 清洗 Query 代码 (防止 Query 里的 docstring 泄露)
            # 注意：对于 Test set，我们要尽量去除可能存在的干扰，虽然生成的代码通常没有 docstring，但安全第一
            query_cleaned = clean_python_code(raw_query)

            # C. BM25 检索
            tokenized_query = query_cleaned.split()
            scores = bm25.get_scores(tokenized_query)
            top_indexes = np.argsort(scores)[-args.top_k:][::-1]

            # D. 组装 Candidates (使用清洗后的数据)
            candidates = []
            for idx in top_indexes:
                train_row = train_data[idx]  # 原始数据（拿元数据）
                cleaned_code_cand = train_corpus_cleaned[idx]  # 清洗后的代码（拿代码）
                score = scores[idx]

                # 处理 Docstring Tokens -> Summary
                tokens = train_row.get('docstring_tokens', [])
                if not tokens and train_row.get('docstring'):
                    tokens = train_row['docstring'].split()
                summary_str = makestr(tokens)

                cand_obj = {
                    "code": cleaned_code_cand,  # 关键：这里存的是没有 docstring 的代码
                    "repo": train_row.get('repo', ''),
                    "path": train_row.get('path', ''),
                    "func_name": train_row.get('func_name', ''),
                    "language": "python",
                    "docstring": train_row.get('docstring', ''),  # 保留原始 docstring 备查
                    "summary": summary_str,  # 关键：新增 summary 字段
                    "idx": int(idx),
                    "bm25_score": float(score)
                }
                candidates.append(cand_obj)

            # E. 组装最终输出对象
            output_obj = test_item.copy()

            # 关键修改：将输出的 code 字段强制替换为清洗后的 Python 代码
            # 这样下游模型读取 'code' 字段时，读到的是干净的 Python 代码
            output_obj['code'] = query_cleaned
            output_obj['retrieved_candidates'] = candidates

            f_out.write(json.dumps(output_obj, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="BM25 retrieval over a CSN Python corpus for each LRPL's "
                    "best Python-pivot translation (see paper §3.3)."
    )
    parser.add_argument("--train_file", type=str,
                        default="./data/CSN/python/train.jsonl",
                        help="Path to the CSN Python training data (BM25 corpus).")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'r', 'racket'],
                        help="Languages whose best Python pivots will be queried.")
    parser.add_argument("--test_template", type=str,
                        default="./data/LowData/{lang}/trans_qwen3th/{lang}_python_best_candidate_vllm_B0.5_S0.5.jsonl",
                        help="Per-lang query JSONL template (output of "
                             "translate_find_bestCode_from_back_vllm.py). "
                             "Use {lang} placeholder.")
    parser.add_argument("--output_template", type=str,
                        default="./data/LowData/{lang}/{lang}_BM25_results.jsonl",
                        help="Per-lang output JSONL template. Use {lang} placeholder.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of BM25 candidates to keep per query.")
    args = parser.parse_args()

    for lang in args.langs:
        print(f"\n{'=' * 30}\nProcessing Language: {lang}\n{'=' * 30}")

        # Per-lang argparse-compatible adapter
        class _A: pass
        a = _A()
        a.train_file  = args.train_file
        a.test_file   = args.test_template.format(lang=lang)
        a.output_file = args.output_template.format(lang=lang)
        a.top_k       = args.top_k

        out_dir = os.path.dirname(a.output_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        run_retrieval(a)


if __name__ == "__main__":
    main()