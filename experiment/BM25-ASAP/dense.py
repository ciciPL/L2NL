import json

import numpy as np
from sentence_transformers import SentenceTransformer # 用于将文本转换为向量
import faiss
# --- 1. 选择/加载编码器模型 ---
# RaxCS 使用的是经过对比学习微调的 CodeBERT (x-encoder) [cite: 145]。
# 对于示例，我们使用一个 SentenceTransformer 模型。
# 模型的选择对嵌入质量至关重要。您可以替换为其他代码专用或更强大的模型。
# 'sentence-transformers/all-MiniLM-L6-v2' 是一个通用但较小的模型。
# 如果需要代码领域更强的嵌入，可以考虑 'jinaai/jina-embeddings-v2-base-zh' (如果您的代码是中文注释混合的)
# 或者其他专门为代码训练的嵌入模型（如 CodeBERT 本身，但需要自行实现 pooling）。
try:
    # 尝试加载一个通用但常用的模型，或者替换为您自己的 CodeBERT/Qwen 嵌入模型
    # 注意：如果您的 CodeBERT/Qwen 模型不能直接用 SentenceTransformer 加载，
    #      您需要手动加载 AutoModel 和 AutoTokenizer，并实现自己的 pooling 逻辑来生成句子嵌入。
    #      RaxCS 的 x-encoder 是 CodeBERT [cite: 145]。
    encoder_model = SentenceTransformer("../../model/sentence-transformers/all-MiniLM-L6-v2")
    encoder_model.eval() # 设置为评估模式
except Exception as e:
    print(f"加载 SentenceTransformer 模型失败: {e}")
    print("请确保模型路径正确，或者尝试其他预训练模型名称。")
    print("无法进行稠密检索，程序退出。")
    exit()

# --- 2. 语料库向量化 ---
# 加载 CSN 数据（与之前 BM25 示例相同的方式）
corpus_data = []              # 存储原始 JSON 数据
corpus_raw_code_strings = []  # 存储原始代码字符串，用于编码

# <<<<<<< 替换为您的实际 CSN Python 训练数据文件路径 >>>>>>>
data_file_path = '../../dataset/CSN/python/train.jsonl'

try:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line)
                if entry.get("language") == "python" and "code" in entry:
                    corpus_data.append(entry)
                    corpus_raw_code_strings.append(entry["code"]) # 使用原始代码字符串进行稠密嵌入
            except json.JSONDecodeError:
                print(f"跳过第 {line_num + 1} 行，因为 JSON 解析错误。")
except FileNotFoundError:
    print(f"错误：文件 '{data_file_path}' 未找到。无法进行稠密检索。")
    exit()

if not corpus_raw_code_strings:
    print("语料库为空，无法进行向量化。请检查数据加载。")
    exit()

print(f"正在将 {len(corpus_raw_code_strings)} 条代码文档向量化 (这可能需要一些时间)...")
# SentenceTransformer.encode 默认将文本转换为 L2 归一化的向量
# corpus_embeddings = encoder_model.encode(corpus_raw_code_strings, convert_to_tensor=True, show_progress_bar=True)
# corpus_embeddings = corpus_embeddings.cpu().numpy() # 转移到 CPU 并转换为 numpy 数组
# np.save('csn_python_corpus_embeddings.npy', corpus_embeddings)
# print(f"语料库向量化完成。向量维度: {corpus_embeddings.shape[1]}")
#todo 加载已有embeddings
corpus_embeddings = np.load('csn_python_corpus_embeddings.npy')

# --- 3. 向量索引 - 使用 FAISS ---
# RaxCS 明确指出使用 FAISS 库进行索引 [cite: 196]。
embedding_dim = corpus_embeddings.shape[1]

# 创建一个 FAISS 索引。IndexFlatL2 使用 L2 距离（欧几里得距离）。
# 如果您的嵌入向量是 L2 归一化的，那么 L2 距离与余弦相似度密切相关（L2_dist^2 = 2 - 2*cosine_sim）。
# RaxCS 使用点积来衡量相似性 [cite: 192]。对于归一化向量，点积就是余弦相似度。
# faiss_index = faiss.IndexFlatL2(embedding_dim)
# faiss_index.add(corpus_embeddings.astype('float32')) # FAISS 期望 float32 类型
# faiss.write_index(faiss_index, 'csn_python_faiss_index.bin')
#todo 加载已有index
faiss_index = faiss.read_index('csn_python_faiss_index.bin')
print(f"FAISS 索引构建完成。索引大小: {faiss_index.ntotal}")

# --- 4. 准备查询 ---
query_original_string = """def count_in_layer(layer, number):   return sum(len(list(filter(lambda x: x == number, row))))  def part_one(input):   width = len(unlist(input[0]))  height = len(input)    n_twos = count_in_layer(input, 2)  n_threes = count_in_layer(input, 3)    return n_twos * width * height if n_twos > n_threes else n_threes * width * height
"""

# --- 5. 查询向量化 ---
print("\n正在将查询代码向量化...")
query_embedding = encoder_model.encode([query_original_string], convert_to_tensor=True)
query_embedding = query_embedding.cpu().numpy() # 转移到 CPU 并转换为 numpy 数组

# --- 6. 相似性搜索 ---
k = 5 # 检索 Top K 个结果
# search 方法返回 (距离, 索引)
distances, indices = faiss_index.search(query_embedding.astype('float32'), k)

print("\n--- 最相关的 5 条代码文档 (来自稠密检索) ---")
for i in range(k):
    doc_index = indices[0][i] # indices 是一个 (num_queries, k) 的数组
    distance = distances[0][i] # distances 也是一个 (num_queries, k) 的数组

    # 将 L2 距离转换为余弦相似度（如果向量是 L2 归一化的）
    # Cosine Similarity = 1 - (L2_distance^2 / 2)
    similarity_score = 1 - (distance**2 / 2)

    retrieved_data = corpus_data[doc_index]

    print(f"\n--- 文档 {doc_index + 1} (余弦相似度: {similarity_score:.4f}, L2 距离: {distance:.4f}) ---")
    print(f"仓库: {retrieved_data.get('repo')}, 路径: {retrieved_data.get('path')}")
    print(f"函数名: {retrieved_data.get('func_name')}")
    print(f"代码片段 (前200字符):\n{retrieved_data.get('code', '')[:200]}...")
    print(f"Docstring 片段 (前20个token):\n{retrieved_data.get('docstring_tokens', '')[:20]}...")