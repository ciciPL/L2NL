from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", device_map="auto",
                                             trust_remote_code=True)


def extract_summary(full_response):
    """精准提取 Summary: 和 Core Logic: 之间的内容"""
    # 找到 Summary: 的起始位置
    summary_start = full_response.find("Summary:")
    if summary_start == -1:
        return None  # 如果没有 Summary:，返回 None

    # 找到 Core Logic: 的起始位置
    core_logic_start = full_response.find("Core Logic:")
    if core_logic_start == -1:
        return None  # 如果没有 Core Logic:，返回 None

    # 提取 Summary: 之后、Core Logic: 之前的内容
    summary = full_response[summary_start + len("Summary:"):core_logic_start].strip()

    # 清理可能的额外换行或空格
    summary = ' '.join(summary.split())  # 合并多余的空格和换行
    return summary


def generate_summary(code):
    # system_content = """You are a code summarizer from code. Output only summary , no comment, no explanations, no examples. You must generate summary with
    #             1. response: summary (<=25 words)"""

    prompt = f"""Generate a ONE-LINE summary (≤25 words) for this code:
Code: {code}
Summary:"""
    # messages = [
    #     {"role": "system",
    #      "content": system_content},
    #     {"role": "user", "content": prompt}
    # ]
    # text = tokenizer.apply_chat_template(
    #     prompt,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature = 0.1,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()


def process_file(input_file, output_file):
    """处理输入文件并生成摘要"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if ':' in line:
                index, code = line.split(':', 1)
                code = code.strip()
                print(f"Processing index {index}...")

                try:
                    summary = generate_summary(code)
                    # summary = extract_summary(summary)
                    outfile.write(f"{index}: {summary}\n")
                    print(f"✅ Success: {index}")
                except Exception as e:
                    error_msg = f"❌ Error processing {index}: {str(e)}"
                    outfile.write(f"{index}: {error_msg}\n")
                    print(error_msg)


if __name__ == "__main__":
    input_file = "dataset/racket/code_rkt.txt"  # 输入文件路径
    output_file = "dataset/rkt_result/rkt_2_NL_1199.txt"  # 输出文件路径

    print("Starting code summarization...")
    process_file(input_file, output_file)
    print(f"Summarization complete! Results saved to {output_file}")
