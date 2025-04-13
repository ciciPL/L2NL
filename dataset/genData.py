# from datasets import Dataset
# from transformers import pipeline
#
# ds = Dataset.from_parquet("racket/racket-00000-of-00001.parquet")
# print(ds)
# doc =[]
# code = []
# for item in ds:
#     content = item['content']
#     str_doc =''
#     str_code =''
#     for line in content.splitlines():
#         if line.startswith(';;'):
#             str_doc = str_doc + str(line).replace(';;', '')
#         else:
#             str_code = str_code + str(line).replace('#lang racket', '')
#     doc.append(str_doc)
#     code.append(str_code)
#
# # 转换注释列表为带有索引的格式
# indexed_comments = []
# for index, comment in enumerate(doc, start=1):
#     indexed_comments.append(f"{index}: {comment}")
# comments_text = '\n'.join(indexed_comments)
#
# # 转换代码列表为带有索引的格式
# indexed_code = []
# for index, line in enumerate(code, start=1):
#     indexed_code.append(f"{index}: {line}")
# code_text = '\n'.join(indexed_code)
#
# # 写入注释文件
# with open('doc_rkt.txt', 'w', encoding='utf-8') as f:
#     f.write(comments_text)
#
# # 写入代码文件
# with open('code_rkt.txt', 'w', encoding='utf-8') as f:
#     f.write(code_text)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
# 预热GPU（避免首次推理延迟）
# 首次运行前先预加载（只需一次）
# python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained(     'model/Qwen2.5-Coder-1.5B-Instruct',     torch_dtype="auto",     device_map="auto",     attn_implementation="flash_attention_2")"
if torch.cuda.is_available():
    torch.zeros(1).cuda()
model_name = "../model/Qwen2.5-Coder-1.5B-Instruct"
start = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)
end = time.perf_counter()
print("model生成" + f"执行耗时: {end - start:.6f} 秒")
start_t = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_name)
end_t = time.perf_counter()
print("token生成"+f"执行耗时: {end_t - start_t:.6f} 秒")
print(torch.cuda.is_available())
try:
    # 打开注释文件
    with open('doc_rkt.txt', 'r', encoding='utf-8') as doc_file, \
            open('code_rkt.txt', 'r', encoding='utf-8') as code_file:
        for index in range(5):
            doc_line = doc_file.readline().replace(str(index)+':', '')
            code_line = code_file.readline().replace(str(index)+':', '')
            system_content = """You are a code summarizer. Generate ultra-concise summaries with:
            1. summary (<=25 words)
            2. Core logic (<=25 words)

            Rules:
            - Omit explanations
            - Never repeat doc content
            - Max 3 lines total"""

            prompt = f"""CODE: {code_line}
            DOC: {doc_line}"""
            messages = [
                    {"role": "system",
                     "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=80,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.2,
                num_beams=2
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print("第{}生成: {}".format(index, response))

except FileNotFoundError:
    print("错误：文件未找到，请检查文件路径和文件名。")
except Exception as e:
    print(f"发生未知错误：{e}")
