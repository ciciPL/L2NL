import numpy as np
import torch
from rouge_score import rouge_scorer

from unixcoder import UniXcoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UniXcoder("model/microsoft/unixcoder-base")
model.to(device)
context = """
 # <mask0>
function bss_host_nid(host)     if haskey(host, "NID")         return host["NID"]     else         error("Could not find nid for ", host, ". Please make sure the bss_nid_to_host_map is up to date.")     end end
"""
tokens_ids = model.tokenize([context], max_length=512, mode="<encoder-decoder>")
source_ids = torch.tensor(tokens_ids).to(device)

prediction_ids = model.generate(source_ids, decoder_only=False, beam_size=3, max_length=128)
predictions = model.decode(prediction_ids)
predictions = [x.replace("<mask0>", "").strip() for x in predictions[0]]
print(predictions)

reference_summary = "Retrieves NID from BSS host object, errors if not found."
# 初始化 ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
# 计算 ROUGE-1 F1 值
rouge_scores = []
for summary in predictions:
    scores = scorer.score(reference_summary, summary)
    rouge_scores.append(scores['rouge1'].fmeasure)

# 找到 ROUGE-1 F1 值最高的摘要
best_summary_index = np.argmax(rouge_scores)
best_summary = predictions[best_summary_index].replace("\n", "")
print(best_summary)
