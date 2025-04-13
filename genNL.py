from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", device_map="auto",
                                             trust_remote_code=True)


def generate_summary(code):
    """使用模型生成代码摘要"""
    prompt = f"""Please analyze this code and provide a detailed English summary of its functionality, inputs, outputs and key algorithms:

    Code:
    {code}

    Summary:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
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
                    outfile.write(f"{index}: {summary}\n")
                    print(f"✅ Success: {index}")
                except Exception as e:
                    error_msg = f"❌ Error processing {index}: {str(e)}"
                    outfile.write(f"{index}: {error_msg}\n")
                    print(error_msg)


if __name__ == "__main__":
    input_file = "racket_codes.txt"  # 输入文件路径
    output_file = "evaluate/racket_codes_summaries.txt"  # 输出文件路径

    print("Starting code summarization...")
    process_file(input_file, output_file)
    print(f"Summarization complete! Results saved to {output_file}")