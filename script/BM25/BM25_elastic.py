import json
import re

from elasticsearch import Elasticsearch, helpers  # 导入 Elasticsearch 客户端和 helpers 用于批量操作


# --- 0. 定义 basic_code_tokenizer (保持不变，用于处理查询字符串) ---
# 注意：在 Elasticsearch 中，实际的 BM25 分词是由 ES 的分析器完成的。
#       这个 tokenizer 在这里主要用于确保查询字符串格式是符合预期的。
def basic_code_tokenizer(code_string):
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\b\w+\b|[^\s\w]', code_string, re.UNICODE)
    return [token for token in tokens if token.strip()]


def BM25_ELASTIC(query_original_string):
    # --- 1. 连接 Elasticsearch ---
    # <<<<<<< 请根据您的 Elasticsearch 实例配置修改 >>>>>>>
    # 默认本地 HTTP 连接，端口 9200
    es = Elasticsearch(
        [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],  # 正确：连接详情作为字典在列表中
    )

    # 尝试连接 Elasticsearch
    try:
        if not es.ping():
            raise ValueError("Elasticsearch 连接失败！请确保 Elasticsearch 服务正在运行。")
        print("成功连接到 Elasticsearch！")
    except Exception as e:
        print(f"连接 Elasticsearch 时发生错误: {e}")
        exit()

    # --- 2. 定义索引名称 ---
    INDEX_NAME = 'csn_python_code_bm25'  # 建议使用一个新名称，避免与旧索引冲突

    # --- 3. 索引语料库 (这是一次性设置步骤) ---
    # 您通常只需要运行此部分代码一次，以将您的数据填充到 Elasticsearch 索引中。
    # 索引创建后，您可以注释掉此代码块，下次运行时直接跳过索引步骤。
    print(f"\n--- 检查并索引数据到 Elasticsearch 索引: {INDEX_NAME} ---")

    if not es.indices.exists(index=INDEX_NAME):
        print(f"索引 '{INDEX_NAME}' 不存在，正在创建...")
        # 定义索引映射 (Mapping)
        # 'code' 字段将被设置为 'text' 类型，Elasticsearch 会对其进行默认分析（包括分词）
        # 'code_tokens' 字段在这里设置为 'keyword' 类型，表示它将作为完整的一个值存储，不会被分词，
        #   因此在本示例中，我们主要通过 'code' 字段进行 BM25 检索。
        mapping = {
            "mappings": {
                "properties": {
                    "repo": {"type": "keyword"},
                    "path": {"type": "keyword"},
                    "func_name": {"type": "keyword"},
                    "original_string": {"type": "text"},
                    "language": {"type": "keyword"},
                    "code": {"type": "text"},  # 此字段将被 ES 的默认分析器分析，用于 BM25 检索
                    "code_tokens": {"type": "keyword"},  # 存储为关键字，不参与 BM25 文本分析
                    "docstring": {"type": "text"},
                    "docstring_tokens": {"type": "keyword"},
                    "sha": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "partition": {"type": "keyword"},
                }
            }
        }
        try:
            es.indices.create(index=INDEX_NAME, body=mapping)
            print(f"索引 '{INDEX_NAME}' 创建成功。")
        except Exception as e:
            print(f"创建索引失败: {e}")
            exit()

        # --- 加载并准备语料库数据以进行索引 ---
        corpus_for_es_indexing = []
        # <<<<<<< 请替换为您的实际 CSN Python 训练数据文件路径 >>>>>>>
        data_file_path = '../../dataset/CSN/python/train.jsonl'

        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        # 仅索引 Python 语言的代码，并且确保有 'code' 字段
                        if entry.get("language") == "python" and "code" in entry:
                            # Elasticsearch 文档需要一个唯一的 ID
                            doc_id = f"python_doc_{line_num}"
                            action = {
                                "_index": INDEX_NAME,
                                "_id": doc_id,
                                "_source": entry  # 将整个 JSON 条目作为源文档存储
                            }
                            corpus_for_es_indexing.append(action)
                    except json.JSONDecodeError:
                        print(f"跳过第 {line_num + 1} 行，因为 JSON 解析错误。")

            if corpus_for_es_indexing:
                print(f"正在索引 {len(corpus_for_es_indexing)} 条文档到 Elasticsearch 中...")
                # 使用 helpers.bulk 进行高效批量索引
                success, failed = helpers.bulk(es, corpus_for_es_indexing, chunk_size=1000)
                print(f"成功索引 {success} 条文档，{len(failed)} 条失败。")
                es.indices.refresh(index=INDEX_NAME)  # 刷新索引，使文档可搜索
            else:
                print("未找到符合条件的 Python 文档进行索引。请检查数据加载路径和筛选条件。")

        except FileNotFoundError:
            print(f"错误：文件 '{data_file_path}' 在索引阶段未找到。请确保路径正确。")
            exit()
    else:
        print(f"索引 '{INDEX_NAME}' 已存在，跳过索引步骤。")

    # --- 4. 准备查询：使用原始 Python 代码字符串 ---
    # 这是您想要进行检索的完整 Python 代码字符串
    # query_original_string = """def count_in_layer(layer, number):   return sum(len(list(filter(lambda x: x == number, row))))  def part_one(input):   width = len(unlist(input[0]))  height = len(input)    n_twos = count_in_layer(input, 2)  n_threes = count_in_layer(input, 3)    return n_twos * width * height if n_twos > n_threes else n_threes * width * height
    # """

    # --- 5. 执行 Elasticsearch 搜索查询 ---
    # Elasticsearch 的 'match' 查询在 'text' 字段上默认使用 BM25 算法。
    # Elasticsearch 会自动对 query_original_string 进行分词，并与索引中的 'code' 字段匹配。
    print("\n--- 在 Elasticsearch 中执行 BM25 搜索 ---")
    search_body = {
        "query": {
            "match": {
                "code": query_original_string  # 查询 'code' 字段
            }
        },
        "size": 3  # 请求返回最相关的 3 条结果
    }

    try:
        response = es.search(index=INDEX_NAME, body=search_body)

        print("\n--- 最相关的 5 条代码文档 (来自 Elasticsearch) ---")
        if response['hits']['hits']:
            for i, hit in enumerate(response['hits']['hits']):
                score = hit['_score']  # 获取 BM25 得分
                retrieved_data = hit['_source']  # 获取原始文档数据

                print(f"\n--- 文档 {i + 1} (得分: {score:.4f}) ---")
                print(f"仓库: {retrieved_data.get('repo')}, 路径: {retrieved_data.get('path')}")
                print(f"函数名: {retrieved_data.get('func_name')}")
                print(f"代码片段 (前200字符):\n{retrieved_data.get('code', '')[:200]}...")
                print(f"Docstring 片段 (前20个token):\n{retrieved_data.get('docstring_tokens', '')[:20]}...")
                return retrieved_data
        else:
            print("未找到相关文档。请检查索引是否已正确填充。")

    except Exception as e:
        print(f"Elasticsearch 搜索失败: {e}")
        print("请确保 Elasticsearch 服务正在运行，并且数据已经成功索引。")
