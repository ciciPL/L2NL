import json
import re
import argparse
import os
from tqdm import tqdm


def clean_candidate_code(code, docstring=""):
    """
    仅用于清洗 retrieved_candidates 中的代码
    去除 \"\"\" 和 ''' 类型的 docstring
    """
    if not code:
        return ""

    # 2. 移除 \"\"\"...\"\"\" 类型 docstring (支持多行、缩进)
    # flags=re.MULTILINE 确保 ^ 能匹配每一行的开头
    code = re.sub(r'^\s*"""[\s\S]*?"""', '', code, flags=re.MULTILINE)

    # 3. 移除 '''...''' 类型 docstring
    code = re.sub(r"^\s*'''[\s\S]*?'''", '', code, flags=re.MULTILINE)

    # 4. 兜底替换：如果提供了原始 docstring 文本，尝试直接替换移除
    if docstring:
        code = code.replace(docstring, "")

    return code.strip()


def makestr(lst):
    """将 docstring_tokens 拼接成 summary"""
    if not lst:
        return ""
    p = ""
    for w in lst:
        p = p + w + " "
    return p.strip()


def process_data(input_file, output_file):
    print(f"Reading raw data from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        # 使用 tqdm 显示进度
        for line in tqdm(f_in, desc="Processing"):
            if not line.strip():
                continue

            # 1. 加载原始数据（保留所有字段）
            item = json.loads(line)

            # --- Top Level 处理 ---
            # 确保 'code' 字段存在，并赋值为 best_python_code
            if 'best_python_code' in item:
                item['code'] = item['best_python_code']

            # --- Candidates 处理 ---
            if 'retrieved_candidates' in item:
                cleaned_candidates = []
                for cand in item['retrieved_candidates']:
                    # 复制一份 candidate 对象，防止引用问题，同时保留 metadata
                    new_cand = cand.copy()

                    # A. 生成 Summary
                    tokens = cand.get('docstring_tokens', [])
                    raw_doc = cand.get('docstring', '')

                    # 兜底：如果没有 tokens 但有 docstring，尝试 split
                    if not tokens and raw_doc:
                        tokens = raw_doc.split()

                    summary_str = makestr(tokens)

                    # B. 清洗 Code (去除 docstring)
                    raw_cand_code = cand.get('code', '')
                    clean_code = clean_candidate_code(raw_cand_code, raw_doc)

                    # C. 更新字段
                    new_cand['code'] = clean_code  # 【修改】清洗后的代码
                    new_cand['summary'] = summary_str  # 【新增】摘要字段

                    cleaned_candidates.append(new_cand)

                # 将处理完的列表放回去
                item['retrieved_candidates'] = cleaned_candidates

            # 写入新文件
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Saved processed data to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Clean docstrings from BM25-retrieved pivot-Python candidates."
    )
    parser.add_argument("--data_root", default="./data/LowData",
                        help="Root directory containing per-language BM25 result jsonl files")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'r', 'racket'],
                        help="Languages to process")
    args = parser.parse_args()

    LANGUAGES = args.langs
    for LANG in LANGUAGES:
        # Input: raw BM25-retrieved jsonl
        INPUT_FILE = f"{args.data_root}/{LANG}/{LANG}_BM25_results.jsonl"
        # Output: docstring-stripped jsonl
        OUTPUT_FILE = f"{args.data_root}/{LANG}/{LANG}_BM25_results_CLEANED.jsonl"

        # 确保目录存在
        out_dir = os.path.dirname(OUTPUT_FILE)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        process_data(INPUT_FILE, OUTPUT_FILE)