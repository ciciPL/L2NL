# import torch
# from unixcoder import UniXcoder
# import random
# import numpy as np
# from rouge_score import rouge_scorer
# from tqdm import tqdm  # Added tqdm for progress bar
#
# # 设置随机种子
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = UniXcoder("model/microsoft/unixcoder-base")
# model.to(device)
# code_path = "code.txt"
# out_path = "result_julia_nl_unx-base.txt"
# model.config.is_decoder = True
#
# # 初始化 ROUGE scorer
# scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
#
# # 假设你有一些参考摘要，如果没有，可以先留空
# reference_summary = ""  # 替换成你的参考摘要
# out = open(out_path, "w")
#
# # 首先计算总行数以便创建进度条
# with open(code_path, "r") as f:
#     total_lines = sum(1 for _ in f)
#
# # 使用 tqdm 创建进度条
# with open(code_path, "r") as f:
#     for line in tqdm(f, total=total_lines, desc="Generating Code Summaries"):
#         line = line.strip()  # 移除行首尾的空白字符
#         content = '"""\n' + "# <mask0>\n" + line + '"""'  # 构建输入字符串
#
#         # 重新初始化Tokenizer (可能需要根据unixcoder库的具体实现进行调整)
#         model.tokenizer = model.tokenizer.from_pretrained("model/microsoft/unixcoder-base")
#
#         tokens_ids = model.tokenize([content], max_length=512, mode="<encoder-decoder>")
#         source_ids = torch.tensor(tokens_ids).to(device)
#         prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
#         predictions = model.decode(prediction_ids)
#         generated_summaries = [x.replace("<mask0>", "").strip() for x in predictions[0]]
#
#         # 计算 ROUGE-1 F1 值
#         rouge_scores = []
#         for summary in generated_summaries:
#             scores = scorer.score(reference_summary, summary)
#             rouge_scores.append(scores['rouge1'].fmeasure)
#
#         # 找到 ROUGE-1 F1 值最高的摘要
#         best_summary_index = np.argmax(rouge_scores)
#         best_summary = generated_summaries[best_summary_index].replace("\n","")
#
#         index = line.split(":")[0]
#         out.write(index + ":" + best_summary + "\n")
#
# out.close()

import random

import numpy as np
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

from unixcoder import UniXcoder

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 配置参数
batch_size = 16  # 可以根据GPU内存调整
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("model/microsoft/unixcoder-base")
model.to(device)
code_path = "code.txt"
ref_path = 'julia_ref.txt'
out_path = "result_julia_nl_unx-base.txt"

# 初始化 ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

# 假设你有一些参考摘要，如果没有，可以先留空
reference_summary = ""  # 替换成你的参考摘要

# 打开输出文件
out = open(out_path, "w")

# 首先计算总行数以便创建进度条
with open(code_path, "r") as f:
    total_lines = sum(1 for _ in f)


# 批处理函数
def process_batch(batch, ref_batch):
    # 准备批量输入
    batch_contents = []
    for line in batch:
        line = line.strip().split(':')[1]
        content = '"""\n' + "#<mask0>\n" + line + '\n"""'
        print(content)
        batch_contents.append(content)

    # # 重新初始化Tokenizer
    # model.tokenizer = model.tokenizer.from_pretrained("model/microsoft/unixcoder-base")

    # 批量分词并填充
    tokens_ids = model.tokenize(batch_contents, max_length=512, mode="<encoder-decoder>", padding=True)

    # 转换为张量并移动到设备
    source_ids = torch.tensor(tokens_ids).to(device)

    # 生成摘要
    prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
    predictions = model.decode(prediction_ids)

    # 处理每个批次的摘要
    results = []
    for i, pred_list in enumerate(predictions):
        generated_summaries = [x.replace("<mask0>", "").strip() for x in pred_list]
        reference_summary = ref_batch[i]
        # 计算 ROUGE-1 F1 值
        rouge_scores = []
        for summary in generated_summaries:
            scores = scorer.score(reference_summary, summary)
            rouge_scores.append(scores['rouge1'].fmeasure)

        # 找到 ROUGE-1 F1 值最高的摘要
        best_summary_index = np.argmax(rouge_scores)
        best_summary = generated_summaries[best_summary_index]

        # 获取原始行的索引

        original_line = batch[i]
        index = original_line.split(":")[0]

        results.append((index, best_summary))

    return results


# 使用批处理进行处理
with open(code_path, "r") as f:
    with open(ref_path, "r") as ref_in:
        # 读取所有行
        lines = f.readlines()
        ref_lines = ref_in.readlines()
        # 使用 tqdm 创建进度条
        for i in tqdm(range(0, len(lines), batch_size), desc="处理批次"):
            # 获取当前批次
            batch = lines[i:i + batch_size]
            ref_batch = ref_lines[i:i + batch_size]

            # 处理批次
            batch_results = process_batch(batch, ref_batch)

            # 写入结果
            for index, summary in batch_results:
                summary = summary.replace("\n", "")
                out.write(f"{index}\t{summary}\n")

# 关闭文件
out.close()

print("处理完成。结果已保存到", out_path)
