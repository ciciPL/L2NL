# 输入文件路径
import re

from transformers import AutoTokenizer

# input_file = 'lengthRef.txt'
model_name = '../../../model/Qwen2.5-Coder-14B-Instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
str = "Reads bytes from an object, counts them using `struct."
tokens = tokenizer.tokenize(str)
print(len(tokens))
# 清除'xxx',且保留15<长度<25的
# with open(input_file, 'r', encoding='utf-8') as f:
#     with open('fixedRef.txt','w',encoding='utf-8') as w:
#         for line in f:
#             nl = line.strip().split(':')[1]
#             index = line.strip().split(':')[0]
#             match = re.search(r"`(.*?)` ", nl)
#             if match:
#                 found_word = re.search(r"`(.*?)` ", nl).group(1)
#                 tokens = tokenizer.tokenize(nl.replace('`'+found_word+'` ', ''))
#             else:
#                 tokens = tokenizer.tokenize(nl)
#             if 15 < len(tokens) < 25:
#                 token_ids = tokenizer.convert_tokens_to_ids(tokens)
#                 decoded_sentence = tokenizer.decode(token_ids)
#                 if '.' in decoded_sentence:
#                     w.write(index + ':' + decoded_sentence.split('.')[0] + '.\n')
# print(decoded_sentence)
# refLen.append(index + ':' + str(len(nl)) + '\n')

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
