import json
import re

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '../../model/deepseek-coder-1.3b-instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)


# model = PeftModel.from_pretrained(model, '../../../finetune/ds-coder/output_dir_ali6k')


# 2. code:{retrieved_data[1]['code']} summary:{retrieved_data[1]['docstring']}
# 3. code:{retrieved_data[2]['code']} summary:{retrieved_data[2]['docstring']}

# def prompt_generate(code_list, index_list):
#     prompts = []
#     datas = []
#     with open('../r_result/r_python_bm25_results.jsonl', 'r') as f:
#         for line in f:
#             datas.append(line)
#     for index, code in zip(index_list, code_list):
#         examples_top3 = datas[int(index)].get("retrieved_top3")
#         code1 = json.loads(examples_top3[0]).get("code")
#         code2 = json.loads(examples_top3[1]).get("code")
#         code3 = json.loads(examples_top3[2]).get("code")
#         nl1 = json.loads(examples_top3[0]).get("summary")
#         nl2 = json.loads(examples_top3[1]).get("summary")
#         nl3 = json.loads(examples_top3[2]).get("summary")
#         print("code1",code1)
#         print("nl1",nl1)
#         prompt = f"""Given the following three examples:
#         1. code:{code1} summary:{nl1}
#         2. code:{code2} summary:{nl2}
#         3. code:{code3} summary:{nl3}
#         Generate a ONE-LINE summary (≤25 words) for this code:
#         Code: {code}
#         Summary:"""
#         prompts.append(prompt)
#     return prompts
def extract_summary(docstring):
    patterns_to_remove = [
        r':param.*$',  # Remove parameter descriptions
        r':return.*$',  # Remove return value descriptions
        r':type.*$',  # Remove type descriptions
        r':rtype.*$',  # Remove return type descriptions
        r'Args:.*$',  # Remove "Args:" section
        r'Parameters.*$',  # Remove "Parameters:" section
        r'>>>.*$',  # Remove Python console examples
        r'^\s*\*\s.*$',  # Remove bullet points
        r'The return type is.*$',  # Remove return type statements
        r'https?://.*$',  # Remove URLs
        r':?Examples?.*$',  # Remove "Example" section
        r':?CLI Examples?.*$',  # Remove "Example" section
        r'Returns.*$',  # Remove "Returns" section
        r'\\.\\.\\.?.*$',  # Remove ellipses
        r'NOTE.*',  # Remove "NOTE" section
        r'See Also.*$',  # Remove "See Also" section
        r'Links.*$',  # Remove "Links" section
        r'Raises.*$',  # Remove "Raises" section
        r'\\s\\w+:.*$',  # Remove any word before : and after
    ]
    # Combine the patterns into a single regular expression
    pattern = re.compile(r'({})'.format('|'.join(patterns_to_remove)), re.MULTILINE | re.DOTALL)
    # Remove the patterns from the docstring
    summary = pattern.sub('', docstring).strip()
    summary = re.sub('\\s+', ' ', summary)
    return summary


def remove_docstrings(code):
    docstrings = re.findall(r"'''(.*?)'''", code, re.DOTALL) + re.findall(r'"""(.*?)"""', code, re.DOTALL)
    for docstring in docstrings:
        code = code.replace(docstring, '')
    code = re.sub(r'""""""\\s*', '', code, re.DOTALL)
    code = re.sub(r"''''''\\s*", '', code, re.DOTALL)
    return code.replace(r'"""', '').replace(r"'''", '')


def remove_comments(code):
    pattern = r"(\\\".*?\\\"|\\'.*?\\')|(#[^\\r\\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (starting with '#')
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, code)


def prompt_generate(code_list, index_list):
    prompts = []

    # 构建 index -> data 的映射
    data_map = {}
    with open('r_result/r_python_bm25_results.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line.strip())  # 解析每一行为字典
            index = item.get("index")
            if index is not None:
                data_map[index] = item  # 按原始 index 存储

    for index, code in zip(index_list, code_list):
        # 从 map 中根据 index 获取对应的检索结果
        retrieved_data = data_map.get(str(index))  # 注意：index 是字符串类型
        if not retrieved_data:
            print(f"Warning: No retrieved data found for index {index}")
            continue

        examples_top3 = retrieved_data.get("retrieved_top3", [])
        if len(examples_top3) < 3:
            print(f"Warning: Less than 3 examples found for index {index}")
            continue

        # 提取三个例子的内容,清洗code和nl
        code1 = remove_docstrings(remove_comments(examples_top3[0].get("code")))
        nl1 = extract_summary(examples_top3[0].get("summary"))
        code2 = remove_docstrings(remove_comments(examples_top3[1].get("code")))
        nl2 = extract_summary(examples_top3[1].get("summary"))
        code3 = remove_docstrings(remove_comments(examples_top3[2].get("code")))
        nl3 = extract_summary(examples_top3[2].get("summary"))

        # 构造 prompt
        prompt = f"""Given the following three examples:
        1. code:
        {code1} 
        summary:{nl1}
        
        2. code:
        {code2} 
        summary:{nl2}
        
        3. code:
        {code3} 
        summary:{nl3}
        
        Generate a ONE-LINE summary (≤25 words) for this code:
        Code: {code}
        Summary:"""
        prompts.append(prompt)
    with open('r_result/prompts_BM25', 'w') as o:
        for index, prompt in enumerate(prompts):
            index += 1
            o.write(str(index) + ":" + prompt + '\n')
    return prompts


def generate_summary_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # 获取每个输入 prompt 在 padding 之前的 token 数量
    # 这是为了之后能准确地分离出新生成的部分
    input_token_lengths = [len(tokenizer.encode(p, truncation=True)) for p in prompts]
    # 或者，更稳健地从 tokenizer 返回的 inputs 中获取（通常通过 attention_mask）
    # input_token_lengths = inputs.attention_mask.sum(dim=1).tolist()

    outputs_tokens = model.generate(
        **inputs,
        max_new_tokens=20,
        min_new_tokens=10,
        temperature=0.1,
        top_p=0.5,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_summaries = []
    for i, output_token_list in enumerate(outputs_tokens):
        prompt_actual_length = inputs.attention_mask[i].sum().item()
        newly_generated_tokens = output_token_list[prompt_actual_length:]
        decoded_summary = tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
        generated_summaries.append(decoded_summary)

    return generated_summaries


def process_file(input_file, output_file, batch_size=4):
    """处理文件（改为分批处理，并加入进度条）"""
    from tqdm import tqdm

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # 获取总行数用于进度条
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # 重置文件指针到开头

        # 使用 tqdm 添加进度条
        lines = tqdm(infile, total=total_lines, desc="Processing code", unit="line")

        # 缓存批次数据
        batch = []
        for line in lines:
            if ':' in line:
                index, code = line.split(':', 1)
                code = code.strip().replace('\\n', '')
                batch.append((index.strip(), code.strip()))

                # 达到批次大小时处理
                if len(batch) >= batch_size:
                    indices, codes = zip(*batch)
                    try:
                        prompts = prompt_generate(codes, indices)
                        print(prompts[:20])
                        summaries = generate_summary_batch(prompts)
                        for idx, summary in zip(indices, summaries):
                            outfile.write(f"{idx}: {summary}\n")
                            # 可选：取消注释下面这行以实时刷新输出
                            # outfile.flush()
                            # print(f"✅ Success: {idx}")
                    except Exception as e:
                        for idx in indices:
                            error_msg = f"❌ Error processing {idx}: {str(e)}"
                            outfile.write(f"{idx}: {error_msg}\n")
                            # outfile.flush()
                            # print(error_msg)
                    batch = []

        # 处理剩余不足一个批次的数据
        if batch:
            indices, codes = zip(*batch)
            try:
                prompts = prompt_generate(codes, indices)
                print(prompts[:20])
                summaries = generate_summary_batch(prompts)
                for idx, summary in zip(indices, summaries):
                    outfile.write(f"{idx}: {summary}\n")
                    # outfile.flush()
                    # print(f"✅ Success: {idx}")
            except Exception as e:
                for idx in indices:
                    error_msg = f"❌ Error processing {idx}: {str(e)}"
                    outfile.write(f"{idx}: {error_msg}\n")
                    # outfile.flush()
                    # print(error_msg)


if __name__ == "__main__":
    # input_file = "../../../dataset/r/code.txt"
    input_file = "r_result/r_2_python_clean.txt"
    output_file = "r_result/r_2_python_2_NL_BM25_top3_cleanPrompts.txt"
    batch_size = 4  # 可调整批大小

    print("Starting code summarization...")
    process_file(input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")
