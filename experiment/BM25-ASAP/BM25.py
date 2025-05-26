import json
import re
from rank_bm25 import BM25Okapi


# --- 0. 定义模拟 CSN 分词的函数 ---
def basic_code_tokenizer(code_string):
    """
    一个简单的分词器，尝试模拟 CSN 对代码的分词方式。
    CSN 的 code_tokens 通常是将单词、数字、字符串和常见标点符号/操作符分开。
    """
    # 匹配模式：
    # 1. 匹配双引号字符串 (e.g., "hello world")
    # 2. 匹配单引号字符串 (e.g., 'hello world')
    # 3. 匹配单词（字母、数字、下划线组成的序列）
    # 4. 匹配单个非空白、非字母数字、非下划线的字符（通常是标点和操作符）
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\b\w+\b|[^\s\w]', code_string, re.UNICODE)

    # 过滤掉 re.findall 可能产生的空字符串
    return [token for token in tokens if token.strip()]


# --- 1. 准备语料库：加载 CSN 数据并直接使用其 'code_tokens' 字段 ---
corpus_data = []  # 存储原始的 JSON 数据
corpus_tokens = []  # 存储 CSN 原始的 code_tokens 列表

# 假设您的 CSN Python 数据文件是 'csn_python_data.jsonl'
# <<<<<<< 请替换为您的实际 CSN Python 数据文件路径 >>>>>>>
data_file_path = '../../dataset/CSN/python/train.jsonl'

try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # 确保是 Python 语言并且包含 'code_tokens' 字段
            if entry.get("language") == "python" and "code_tokens" in entry:
                corpus_data.append(entry)
                corpus_tokens.append(entry["code_tokens"])  # 直接使用 CSN 提供的 code_tokens
except FileNotFoundError:
    print(f"错误：文件 '{data_file_path}' 未找到。请确保文件存在并路径正确。")
    exit()

print(f"成功加载 {len(corpus_data)} 条 Python 代码文档作为 BM25 语料库（使用 CSN 原始 code_tokens）。")

# --- 2. 初始化 BM25 模型 ---
bm25 = BM25Okapi(corpus_tokens)

# --- 3. 准备查询：使用原始 Python 代码字符串 ---
# 这是您想要进行检索的完整 Python 代码字符串
query_original_string = """def count_in_layer(layer, number):	return sum(len(list(filter(lambda x: x == number, row))))	def part_one(input):	width = len(unlist(input[0]))	height = len(input)	n_twos = count_in_layer(input, 2)	n_threes = count_in_layer(input, 3)	return n_twos * width * height if n_twos > n_threes else n_threes * width * height
"""

# --- 4. 关键：用您自定义的 basic_code_tokenizer 对查询代码进行分词 ---
query_tokens = basic_code_tokenizer(query_original_string)
print(f"\n查询代码的 token 列表 (由 basic_code_tokenizer 生成，前20个): {query_tokens[:20]}...")

# --- 5. 获取相关性得分 ---
doc_scores = bm25.get_scores(query_tokens)

# --- 6. 排序并显示结果 ---
ranked_docs_indices = doc_scores.argsort()[::-1]

print("\n--- 最相关的 5 条代码文档 ---")
if corpus_data:  # 确保语料库不为空
    for i in range(min(5, len(ranked_docs_indices))):
        doc_index = ranked_docs_indices[i]
        score = doc_scores[doc_index]
        retrieved_data = corpus_data[doc_index]

        print(f"\n--- 文档 {doc_index + 1} (得分: {score:.4f}) ---")
        print(f"仓库: {retrieved_data.get('repo')}, 路径: {retrieved_data.get('path')}")
        print(f"函数名: {retrieved_data.get('func_name')}")
        print(f"代码片段 (CSN code_tokens 前20个):\n{retrieved_data.get('code_tokens', '')[:20]}...")
        # 为了对比，可以打印检索到的原始代码字符串
        print(f"原始代码片段 (前200字符):\n{retrieved_data.get('code', '')[:200]}...")
else:
    print("语料库为空，无法进行检索。请检查数据加载路径和筛选条件。")