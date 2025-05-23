from peft import PeftModel
from sympy.physics.units import temperature
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = '../../../model/deepseek-ai/deepseek-coder-1.3b-instruct'
# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0},
                                             trust_remote_code=True)
# model = PeftModel.from_pretrained(model, '../../../finetune/ds-coder/output_dir_ali6k')


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
        max_new_tokens=25,
        min_new_tokens=10,
        temperature=0.1,
        top_p=0.5,
        do_sample=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
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
                code  =  code.strip().replace('\\n','')
                print(code)
                batch.append((index.strip(),code.strip() ))

                # 达到批次大小时处理
                if len(batch) >= batch_size:
                    indices, codes = zip(*batch)
                    try:
                        summaries = generate_summary_batch(codes)
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
                summaries = generate_summary_batch(codes)
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
    input_file = "../../../dataset/lua/code.txt"
    # input_file ="../r_result/r_2_python_clean.txt"
    output_file = "../lua_result/lua_2_NL.txt"
    batch_size = 4  # 可调整批大小

    print("Starting code summarization...")
    process_file(input_file, output_file, batch_size)
    print(f"Summarization complete! Results saved to {output_file}")
