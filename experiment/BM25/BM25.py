import json
import re
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, data_file_path='../../../dataset/CSN/python/train.jsonl'):
        """
        在初始化时加载数据并构建 BM25 模型。
        """
        print("--- 正在初始化 BM25 Retriever ---")
        self.corpus_data = []
        corpus_tokens = []

        try:
            with open(data_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("language") == "python" and "code_tokens" in entry:
                        self.corpus_data.append(entry)
                        corpus_tokens.append(entry["code_tokens"])
        except FileNotFoundError:
            print(f"错误：文件 '{data_file_path}' 未找到。请确保文件存在并路径正确。")
            raise  # 抛出异常，让调用者知道出错了

        if not corpus_tokens:
            raise ValueError("语料库为空，无法初始化 BM25。请检查数据文件和筛选条件。")

        print(f"成功加载 {len(self.corpus_data)} 条 Python 代码文档。")
        print("正在构建 BM25 模型...")
        self.bm25 = BM25Okapi(corpus_tokens)
        print("--- BM25 Retriever 初始化完成 ---")

    @staticmethod
    def basic_code_tokenizer(code_string):
        """
        静态方法：简单的代码分词器。
        """
        tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\b\w+\b|[^\s\w]', code_string, re.UNICODE)
        return [token for token in tokens if token.strip()]

    def retrieve(self, query_original_string, top_n=3):
        """
        执行单次检索。
        """
        query_tokens = self.basic_code_tokenizer(query_original_string)
        doc_scores = self.bm25.get_scores(query_tokens)
        ranked_docs_indices = doc_scores.argsort()[::-1]

        result = []
        scores = []
        for i in range(min(top_n, len(ranked_docs_indices))):
            doc_index = ranked_docs_indices[i]
            score = doc_scores[doc_index]
            result.append(self.corpus_data[doc_index])
            scores.append(score)
        # 如果需要，可以确保返回 N 个结果，即使不足也用第一个填充（或者返回更少）
        # while len(result) < top_n and len(self.corpus_data) > 0:
        #     result.append(self.corpus_data[0])

        return result, scores
