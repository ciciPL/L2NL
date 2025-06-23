# 输入文件路径
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from langdetect import detect
from transformers import AutoTokenizer

# input_file = 'lengthRef.txt'
model_name = '../../../model/deepseek-coder-1.3b-instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 英文清洗函数
def clean_summary(text):
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去除URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 去除邮箱
    text = re.sub(r'\S+@\S+', '', text)

    # 去除非英文字符和常用标点
    text = re.sub(r'[^a-zA-Z\s' + re.escape(string.punctuation) + ']', '', text)

    # 去除连续多个空格
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # 新增：检查结尾是否是以 '.' 或 ' .' 结尾
    if not text.endswith('.') and not text.endswith(' .'):
        return None

    # 替换 ' .' 为 '.'
    if text.endswith(' .'):
        text = text[:-2] + '.'

    # 特殊字符占比判断
    special_chars = [c for c in text if c in string.punctuation]
    if len(special_chars) / len(text) > 0.05:
        return None

    # 使用 tokenizer 分词
    tokens = tokenizer.tokenize(text)
    if not (10 < len(tokens) < 40):
        return None

    # 可选：语言检测为英文
    try:
        if detect(text) != 'en':
            return None
    except:
        return None

    # 去除全标点或全空格
    if all(c in string.punctuation or c.isspace() for c in text):
        return None

    return text

#统计数据长度分布
def tongji(file_path,title):
    cleaned_data = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                idx_str, text = line.strip().split(':', 1)
                cleaned_data[int(idx_str)] = text.strip()

    # 提取 token 长度
    lengths = []
    for idx, text in cleaned_data.items():
        tokens = tokenizer.tokenize(text)
        lengths.append(len(tokens))

    # 输出统计信息
    print(f"总样本数: {len(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数长度: {np.median(lengths)}")
    print(f"最短长度: {min(lengths)}")
    print(f"最长长度: {max(lengths)}")
    print(f"95% 分位数: {np.percentile(lengths, 95):.0f}")

    # 画图
    bin_edges = np.arange(min(lengths), max(lengths) + 1)  # 每个整数一个 bin

    plt.figure(figsize=(12, 6))
    sns.histplot(lengths, bins=bin_edges, kde=False, color='skyblue', discrete=True)
    plt.title(title, fontsize=14)
    plt.xlabel('Token Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(bin_edges)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    #清洗
    # with open('../Clean_PCSD/train/PCSD_ref_gen_vllm_prompt2th_clean.txt', 'r') as f,\
    #     open('../Clean_PCSD/train/extract_PCSD_ref_gen_vllm_prompt2th_clean.txt','w') as o:
    #     for line in f:
    #         index = line.split(":")[0]
    #         code = line.split(":")[1]
    #         clean_line = clean_summary(code)
    #         if clean_line is not None:
    #             o.write(index+": "+clean_line.strip()+"\n")

    #统计
    tongji("../Clean_PCSD/train/extract_ref.txt","extract_ref.txt")
    #输出特定长度的摘要
    # with open('../Clean_PCSD/train/extract_PCSD_ref_gen_vllm_prompt2th_clean.txt','r', encoding='utf-8') as f:
    #     for line in f:
    #         code = line.split(":")[1]
    #         tokens = tokenizer.tokenize(code)
    #         if 23 < len(tokens) < 26:
    #             print(len(tokens))


# 加入原ref长度合适的加入到LLM生成摘要筛选集中，形成半合成数据
# llmId = []  # 合适的半合成数据ids，用于跳过这些ids，在剩下原始数据集筛选时
# code = {}
# with open('lengthRef.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         index = line.split(':')[0]
#         llmId.append(index)
#         code[index] = line.split(':')[1]
# with open('aliref.txt', 'r', encoding='utf-8') as f:
#     with open('fixedRef.txt', 'w', encoding='utf-8') as w:
#         for line in f:
#             index = line.split(':')[0]
#             if index in llmId:
#                 w.write(index + ':' + code[index])
#                 continue
#             nl = line.strip().split(':')[1]
#             # 去除'xxx'
#             match = re.search(r"`(.*?)` ", nl)
#             if match:
#                 found_word = re.search(r"`(.*?)` ", nl).group(1)
#                 tokens = tokenizer.tokenize(nl.replace('`' + found_word + '` ', ''))
#             else:
#                 tokens = tokenizer.tokenize(nl)
#             if 15 < len(tokens) < 25:
#                 token_ids = tokenizer.convert_tokens_to_ids(tokens)
#                 decoded_sentence = tokenizer.decode(token_ids)
#                 if '.' in decoded_sentence:
#                     w.write(index + ':' + decoded_sentence.strip() + '\n')
