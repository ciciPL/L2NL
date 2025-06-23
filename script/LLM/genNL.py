import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = '../../model/deepseek-ai/deepseek-coder-1.3b-instruct'
model_path = '../../model/deepseek-coder-1.3b-instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0}, attn_implementation="flash_attention_2",
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True)
model = PeftModel.from_pretrained(model, '../../finetune/ds-coder/output_dir_ali6k')

# 2. code:{retrieved_data[1]['code']} summary:{retrieved_data[1]['docstring']}
# 3. code:{retrieved_data[2]['code']} summary:{retrieved_data[2]['docstring']}

def prompt_generate_without_sentence(code_list, index_list):

    prompts=[]
    for code in code_list:
        prompt_without_sentence = f"""
        You are an expert code summarization AI. Your task is to generate a concise, ONE-LINE summary based on the provided CODE.
        
        USER_INPUT_CODE:
        {code}
        """
        prompts.append(prompt_without_sentence)
    return prompts


def prompt_generate_BM25(code_list, index_list):
    prompts = []

    # 构建 index -> data 的映射
    data_map = {}
    with open('../r_result/r_python_bm25_results.jsonl', 'r') as f:
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

        # 提取三个例子的内容
        code1 = examples_top3[0].get("code")
        nl1 = examples_top3[0].get("summary")
        code2 = examples_top3[1].get("code")
        nl2 = examples_top3[1].get("summary")
        code3 = examples_top3[2].get("code")
        nl3 = examples_top3[2].get("summary")

        # 构造 prompt
        prompt = f"""Given the following three examples:
        1. code:{code1} summary:{nl1}
        2. code:{code2} summary:{nl2}
        3. code:{code3} summary:{nl3}
        Generate a ONE-LINE summary (≤25 words) for this code:
        Code: {code}
        Summary:"""
        prompts.append(prompt)

    return prompts

def generate_summary_batch(prompts,code_list):
    """新增批量处理函数（核心逻辑不变）"""
    sentences=[]
    with open('../EASC/output/python/predictions/python_sentences_preds.jsonl','r') as f:
        for line in f:
            data  = json.loads(line)
            cleaned_seqs_pred = data.get("cleaned_seqs_pred")
            sentences.append(cleaned_seqs_pred)
    prompts = [
        f"""Given the following R code snippet and its corresponding Python translation:
        [R Code snippet]:
        {sentence}
    
        [Python Translation]
        {code}
        
        Please generate a concise natural language description of what this code does:
        Summary:"""
        for code,sentence in zip(code_list,sentences)
    ]


    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=25,
        min_new_tokens=10,
        temperature=0.1,
        top_p=0.5,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    return [
        tokenizer.decode(output, skip_special_tokens=True).replace(prompt, "").strip()
        for output, prompt in zip(outputs, prompts)
    ]


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
                        prompts = prompt_generate_without_sentence(codes, indices)
                        summaries = generate_summary_batch(prompts,codes)
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
                prompts = prompt_generate_without_sentence(codes, indices)
                summaries = generate_summary_batch(prompts,codes)
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
    # input_file = "../../dataset/LowData/r/code.txt"
    input_file = "../../experiment/ds-coder-1_3B/r_result/r_2_python_clean.txt"
    output_file = "../../experiment/ds-coder-1_3B/r_result/r_2_python_2_NL_sentences_2th.txt"
    batch_size = 8  # 可调整批大小

    print("Starting code summarization...")
    process_file(input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")
