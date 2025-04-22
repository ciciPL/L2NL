from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm  # 导入进度条库

model_name = '../../../model/deepseek-coder-1.3b-instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True)


def generate_summary_batch(code_list):
    """新增批量处理函数（核心逻辑不变）"""
    prompts = [
        f"""Generate a ONE-LINE summary (≤25 words) for this code:
Code: {code}
Summary:"""
        for code in code_list
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.1,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id
    )

    return [
        tokenizer.decode(output, skip_special_tokens=True).replace(prompt, "").strip()
        for output, prompt in zip(outputs, prompts)
    ]


def process_file(input_file, output_file, batch_size=4):
    """处理文件（添加进度条功能）"""
    # 先计算总行数用于进度条
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # 初始化进度条
        pbar = tqdm(total=total_lines, desc="Processing", unit="lines")

        # 缓存批次数据
        batch = []
        for line in infile:
            if ':' in line:
                index, code = line.split(':', 1)
                batch.append((index.strip(), code.strip()))
                pbar.update(1)  # 更新进度条

                # 达到批次大小时处理
                if len(batch) >= batch_size:
                    indices, codes = zip(*batch)
                    try:
                        summaries = generate_summary_batch(codes)
                        for idx, summary in zip(indices, summaries):
                            outfile.write(f"{idx}: {summary}\n")
                    except Exception as e:
                        for idx in indices:
                            error_msg = f"❌ Error processing {idx}: {str(e)}"
                            outfile.write(f"{idx}: {error_msg}\n")
                    batch = []

        # 处理剩余不足一个批次的数据
        if batch:
            indices, codes = zip(*batch)
            try:
                summaries = generate_summary_batch(codes)
                for idx, summary in zip(indices, summaries):
                    outfile.write(f"{idx}: {summary}\n")
            except Exception as e:
                for idx in indices:
                    error_msg = f"❌ Error processing {idx}: {str(e)}"
                    outfile.write(f"{idx}: {error_msg}\n")

        pbar.close()  # 关闭进度条


if __name__ == "__main__":
    input_file = "../../../dataset/julia/code.txt"
    output_file = "../julia_result/julia_2_NL.txt"
    batch_size = 64  # 可调整批大小

    print("Starting code summarization...")
    process_file(input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")