from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "model/deepseek-coder-1.3b-instruct",
    trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "model/deepseek-coder-1.3b-instruct",
    device_map="auto",
    trust_remote_code=True)


def generate_summary_batch(rkt_code_list, python_code_list):
    """批量生成摘要（核心逻辑不变）"""
    prompts = []
    for rkt_code, python_code in zip(rkt_code_list, python_code_list):
        prompt = f"""
        Generate a concise Racket code summary (≤25 words) following these guidelines:

        [Primary Source]
        Racket Code:
        {rkt_code}

        [Reference Only]
        Python Code (for structural reference only):
        {python_code}

        Generation Rules:
        1. MUST prioritize Racket semantics and terminology (e.g., use 'define' not 'def')
        2. Extract core functionality ONLY from Racket code
        3. May adapt Python's structural patterns (e.g., docstring format) if applicable
        4. MUST NOT exceed 25 words
        5. MUST maintain Racket's functional programming style

        Example Transformation:
        Python: "def add(a,b): return a+b" → Racket: "define (add a b) (+ a b)"
        Summary: "Defines a function that returns the sum of two numbers"

        Final Summary:"""
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.1,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id
    )

    return [tokenizer.decode(output, skip_special_tokens=True).split("Summary:")[-1].strip()
            for output in outputs]


def process_files(rkt_input_file, python_input_file, output_file, batch_size=4):
    """处理两个输入文件并生成摘要"""
    with open(rkt_input_file, 'r', encoding='utf-8') as rkt_file, \
            open(python_input_file, 'r', encoding='utf-8') as py_file, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        # 计算总行数（假设两个文件行数相同）
        total_lines = sum(1 for _ in open(rkt_input_file, 'r', encoding='utf-8'))

        # 准备进度条
        pbar = tqdm(total=total_lines, desc="Processing", unit="lines")

        # 缓存批次数据
        batch_indices = []
        batch_rkt_codes = []
        batch_py_codes = []

        for rkt_line, py_line in zip(rkt_file, py_file):
            # 处理Racket代码行
            if ':' in rkt_line:
                rkt_index, rkt_code = rkt_line.split(':', 1)
                batch_indices.append(rkt_index.strip())
                batch_rkt_codes.append(rkt_code.strip())

            # 处理Python代码行（假设格式相同）
            if ':' in py_line:
                _, py_code = py_line.split(':', 1)
                batch_py_codes.append(py_code.strip())

            # 更新进度条
            pbar.update(1)

            # 达到批次大小时处理
            if len(batch_indices) >= batch_size:
                try:
                    summaries = generate_summary_batch(batch_rkt_codes, batch_py_codes)
                    for idx, summary in zip(batch_indices, summaries):
                        outfile.write(f"{idx}: {summary}\n")
                except Exception as e:
                    for idx in batch_indices:
                        outfile.write(f"{idx}: ❌ Error: {str(e)}\n")

                # 清空批次缓存
                batch_indices = []
                batch_rkt_codes = []
                batch_py_codes = []

        # 处理剩余不足一个批次的数据
        if batch_indices:
            try:
                summaries = generate_summary_batch(batch_rkt_codes, batch_py_codes)
                for idx, summary in zip(batch_indices, summaries):
                    outfile.write(f"{idx}: {summary}\n")
            except Exception as e:
                for idx in batch_indices:
                    outfile.write(f"{idx}: ❌ Error: {str(e)}\n")

        pbar.close()


if __name__ == "__main__":
    rkt_input_file = "dataset/racket/code_rkt.txt"  # Racket代码文件
    python_input_file = "dataset/rkt_result/rkt_2_python_40510.txt"  # Python参考代码文件
    output_file = "dataset/rkt_result/rkt_2_NL_rkt-Python.txt"  # 输出摘要文件
    batch_size = 64  # 可根据GPU内存调整

    print("Starting cross-language code summarization...")
    process_files(rkt_input_file, python_input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")