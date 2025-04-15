from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("../model/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../model/deepseek-coder-1.3b-instruct", device_map={"": 0},
                                             trust_remote_code=True)


def extract_summary(full_response):
    """精准提取 Summary: 和 Core Logic: 之间的内容（完全不变）"""
    summary_start = full_response.find("Summary:")
    if summary_start == -1:
        return None

    core_logic_start = full_response.find("Core Logic:")
    if core_logic_start == -1:
        return None

    summary = full_response[summary_start + len("Summary:"):core_logic_start].strip()
    summary = ' '.join(summary.split())
    return summary


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
    """处理文件（改为分批处理，其他逻辑不变）"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # 缓存批次数据
        batch = []
        for line in infile:
            if ':' in line:
                index, code = line.split(':', 1)
                batch.append((index.strip(), code.strip()))

                # 达到批次大小时处理
                if len(batch) >= batch_size:
                    indices, codes = zip(*batch)
                    try:
                        summaries = generate_summary_batch(codes)
                        for idx, summary in zip(indices, summaries):
                            outfile.write(f"{idx}: {summary}\n")
                            print(f"✅ Success: {idx}")
                    except Exception as e:
                        for idx in indices:
                            error_msg = f"❌ Error processing {idx}: {str(e)}"
                            outfile.write(f"{idx}: {error_msg}\n")
                            print(error_msg)
                    batch = []

        # 处理剩余不足一个批次的数据
        if batch:
            indices, codes = zip(*batch)
            try:
                summaries = generate_summary_batch(codes)
                for idx, summary in zip(indices, summaries):
                    outfile.write(f"{idx}: {summary}\n")
                    print(f"✅ Success: {idx}")
            except Exception as e:
                for idx in indices:
                    error_msg = f"❌ Error processing {idx}: {str(e)}"
                    outfile.write(f"{idx}: {error_msg}\n")
                    print(error_msg)


if __name__ == "__main__":
    input_file = "../../../dataset/racket/code_rkt.txt"
    output_file = "expriment/ds-coder-1_3B/rkt_result/rkt_2_NL_40510.txt"
    batch_size = 32  # 可调整批大小

    print("Starting code summarization...")
    process_file(input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")