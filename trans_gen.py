from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


def load_racket_codes(file_path):
    """加载 Racket 代码文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]


def translate_racket_to_python(racket_code, tokenizer, model, device):
    """优化翻译函数，精确提取Python代码"""
    # 更严格的prompt模板
    prompt = f"""Convert this Racket function to Python equivalent. 
Output ONLY the Python code with proper indentation. 
No explanations, no examples.

Racket:
{racket_code}

genPython:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id
    )

    # 清洗输出
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 正则表达式匹配第一个Python代码块
    import re
    py_code_match = re.search(
        r'^```python\s*(.*?)\s*```|^def\s.*?(?=\n\s*def|\Z)',
        full_text,
        re.DOTALL
    )

    return py_code_match.group(1) if py_code_match else full_text


def translate_and_save(codes, output_file, tokenizer, model, device):
    """带错误处理的翻译保存函数"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, code in enumerate(codes, 1):
            try:
                if ':' in code:
                    parts = code.split(':', 1)
                    index, racket_code = parts[0], parts[1].strip()
                else:
                    index, racket_code = str(idx), code
                print("racket code", racket_code)
                python_code = translate_racket_to_python(racket_code, tokenizer, model, device)
                f.write(f"{index}: {python_code}\n")
                print(f"✅ 成功翻译 #{index}")

            except Exception as e:
                error_msg = f"❌ 翻译失败 #{index}: {str(e)}"
                print(error_msg)
                f.write(f"{index}: {error_msg}\n")


if __name__ == "__main__":
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            padding_side="left",
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            device_map="auto",
            trust_remote_code=True
        ).to(device)  # 确保模型在目标设备上

    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 文件处理
    input_file = "dataset/rkt_result/code_rkt.txt"
    output_file = "dataset/rkt_result/rkt_python_40510_DS_1-3b.txt"

    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        exit(1)

    print("开始翻译...")
    racket_codes = load_racket_codes(input_file)
    translate_and_save(racket_codes, output_file, tokenizer, model, device)
    print(f"翻译完成！结果已保存到 {output_file}")