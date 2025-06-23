import argparse
import json
import sys

from tqdm import tqdm

# 确保 BM25_Retriever.py 在 Python 路径中
# 如果 BM25_Retriever.py 在上一级目录的 BM25 文件夹中
sys.path.append('../../BM25')
try:
    from BM25 import BM25Retriever  # 假设文件名是 BM25.py
except ImportError:
    print("错误: 无法导入 BM25Retriever。请确保 BM25.py 存在且路径正确。")
    sys.exit(1)


def process_code_file(input_file, output_file, bm25_retriever, num_to_process=None):
    """
    读取 code.txt, 执行 BM25 检索, 并将结果保存到 JSON 文件。

    Args:
        input_file (str): 输入的 code.txt 文件路径。
        output_file (str): 输出的 JSON 文件路径。
        bm25_retriever (BM25Retriever): 预先初始化的 BM25 检索器实例。
        num_to_process (int, optional): 要处理的记录数。如果为 None，则处理所有记录。
    """
    results = []
    print(f"正在处理文件: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines_to_read = infile.readlines()

            # 如果设置了 num_to_process，则只取前 N 行
            if num_to_process is not None:
                lines_to_read = lines_to_read[:num_to_process]

            for line in tqdm(lines_to_read, desc="Retrieving BM25"):
                line = line.strip()
                if ':' in line:
                    try:
                        index, code = line.split(':', 1)
                        index = index.strip()
                        # 清理 code，特别是 \n (如果存在)
                        code = code.strip().replace('\\n', ' ')

                        # 执行 BM25 检索 (Top 3)
                        retrieved_docs, retrieved_scores = bm25_retriever.retrieve(code, top_n=3)

                        # 格式化检索结果
                        retrieved_examples = []
                        for doc, score in zip(retrieved_docs, retrieved_scores):
                            retrieved_examples.append({
                                'code': doc.get('code', 'N/A'),
                                'summary': doc.get('docstring', 'N/A'),
                                'score': score
                            })
                        print(retrieved_examples)
                        # 存储结果
                        results.append({
                            'index': index,
                            'original_code': code,
                            'retrieved_top3': retrieved_examples
                        })

                    except ValueError:
                        print(f"警告: 无法解析行: {line}")
                    except Exception as e:
                        print(f"处理行 '{line}' 时发生错误: {e}")
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file}' 未找到。")
        return

    print(f"处理完成，正在将 {len(results)} 条结果写入到: {output_file}")
    # 将结果写入 JSON Lines 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in results:
                outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("文件写入成功！")
    except Exception as e:
        print(f"写入文件 '{output_file}' 时发生错误: {e}")


if __name__ == "__main__":
    # --- 参数设置 ---
    parser = argparse.ArgumentParser(description="使用 BM25 检索代码示例并保存结果。")
    parser.add_argument("--input_file", type=str, default="../ds-coder-1_3B/lua_result/lua_2_python_4081.txt",
                        help="输入的 code.txt 文件路径 (格式: index:code)")
    parser.add_argument("--output_file", type=str,
                        default="../ds-coder-1_3B/lua_result/lua_python_bm25_results.jsonl",
                        help="输出的 JSON Lines 文件路径")
    parser.add_argument("--csn_file", type=str, default='../../dataset/CSN/python/train.jsonl',
                        help="CSN 语料库文件路径 (用于 BM25)")
    parser.add_argument("--num_records", type=int, default=-1,
                        help="要处理的记录数量 (默认: 10，设为 -1 处理所有)")

    args = parser.parse_args()

    num_to_process = args.num_records if args.num_records != -1 else None

    # --- 初始化 BM25 ---
    try:
        bm25_instance = BM25Retriever(data_file_path=args.csn_file)
    except Exception as e:
        print(f"BM25 初始化失败: {e}")
        sys.exit(1)

    # --- 处理文件 ---
    process_code_file(
        input_file=args.input_file,
        output_file=args.output_file,
        bm25_retriever=bm25_instance,
        num_to_process=num_to_process
    )
