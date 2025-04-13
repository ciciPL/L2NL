import subprocess

import evaluate
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score, single_meteor_score
from rouge_score import rouge, rouge_scorer

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 读取参考摘要和模型输出（每行一个样本）
def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line for line in f]

# 加载数据（假设每行对应一个样本，参考和结果一一对应）
ref_path = os.path.join(current_dir,'racket_summaries.txt')  # 参考摘要文件
result_path = os.path.join(current_dir,'racket_python_summaries.txt')  # 模型输出文件

refs = read_lines(ref_path)
results = read_lines(result_path)


ref_dict = {int(line.split('.')[0]): line.split('.')[1] for line in refs}
result_dict = {}
need_list=[]
for line in results:
    parts = line.strip().split(':')
    if len(parts) > 1:
        num = int(parts[0])
        data = parts[1]
        result_dict[num] = data
        need_list.append(num)
    else:
        try:
            int(parts[0])
        except (ValueError, IndexError):
            continue
ref_dict_clen={}
for key in need_list:
    ref_dict_clen[key] = ref_dict[key]
# ---------------------- BLEU-4 计算 ----------------------
# 格式要求：sacrebleu 需要每个样本的参考列表（每个样本可能有多个参考，这里假设单个参考）
# 对文本进行分词（英文示例，中文需替换为中文分词）
def tokenize_bleu(data,isRef):
    result = []
    for i in range(len(data)):
        words = word_tokenize(data[i])
        if isRef:
            result.append([words])
        else:
            result.append(words)
    # print(result)
    return result

ref_list = [value for value in ref_dict_clen.values()]
hys_list = [value for value in result_dict.values()]

# 转换为 sacrebleu 所需的格式（每个样本的参考列表包裹在列表中）
bleu_score = corpus_bleu(tokenize_bleu(ref_list,True), tokenize_bleu(hys_list,False))
# ---------------------- ROUGE-L 计算 ----------------------

# 初始化 ROUGE 计分器
# 指定要计算的 ROUGE 类型，这里选择了 rouge1、rouge2 和 rougeL
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# 输出结果
rouge=0
for i, (ref, pred) in enumerate(zip(ref_list, hys_list)):
    scores = scorer.score(ref, pred)
    # print(f"Sample {i + 1}:")
    for rouge_type, score in scores.items():
        rouge += score.fmeasure
    #     print(f"  {rouge_type}:")
    #     print(f"    Precision: {score.precision}")
    #     print(f"    Recall: {score.recall}")
    #     print(f"    F1-score: {score.fmeasure}")
    # print()
#
# ---------------------- METEOR 计算 ----------------------
# 初始化 METEOR（需指定 METEOR JAR 路径，首次运行会自动下载）
# meteor_jar_path = 'meteor/meteor-1.5.jar'  # 替换为你下载的 meteor-1.5.jar 文件的实际路径
## meteor = evaluate.load("metrics/meteor")
    # print([[ref]])
    # print([hyp])
    # score = meteor.compute(predictions=[hyp], references=[[ref]])
    # print(score)
meteor_scores = []
for hyp, ref in zip(hys_list, ref_list):
    score = single_meteor_score(ref.split(), hyp.split())
    print(score)
    meteor_scores.append(score)
    # score = meteor_score([word_tokenize(ref)],word_tokenize(hyp))
    # meteor_scores += score

print(f"METEOR Score: {sum(meteor_scores)*100/len(meteor_scores):.4f}")
# 输出结果
print(f"BLEU-4 Score: {bleu_score*100:.4f}")
print(f"ROUGE-L Score: {rouge*100/len(refs):.4f}")
# print(f"METEOR Score: {meteor_scores*100/len(refs):.4f}")