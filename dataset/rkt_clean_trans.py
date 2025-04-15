from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("translation_clean.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CleanTranslator:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def initialize(self):
        """初始化模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()

            logger.info(f"模型加载成功: {self.device}")
            return True
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            return False

    #     def generate_prompt(self, racket_code):
    #         """强化prompt指令"""
    #         return f"""Convert this Racket function to Python equivalent.
    # Output ONLY the Python code with proper indentation.
    # STRICTLY follow these rules:
    # 1. No explanations or comments
    # 2. No Note: sections
    # 3. No additional text
    # 4. Only include the function definition
    #
    # Racket:
    # {racket_code}
    # """

    def extract_clean_code(self, generated_text):
        """严格提取纯代码"""
        # 模式1：匹配从def开始到文件结尾的代码
        pattern1 = r'(def\s[\s\S]*?)(?=\n\s*\d+:|\Z)'
        # 模式2：匹配genPython:后的代码块
        pattern2 = r'(?<=genPython:\n)([\s\S]*?)(?=\n\s*(?:Note:|Racket:|$))'

        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                # 移除所有注释和非代码行
                code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
                code = re.sub(r'Note:.*$', '', code, flags=re.MULTILINE)
                # 移除空行并标准化缩进
                lines = [line.rstrip() for line in code.split('\n') if line.strip()]
                return '\n'.join(lines)

        return None

    def translate(self, racket_code):
        system_content = """you are a translater from racket to python, Output ONLY the Python code with proper indentation. No explanations, no examples , and you must Convert this Racket function to Python equivalent with:
            1. python 
            """

        prompt = f"""RACKET_CODE: {racket_code} """
        messages = [
            {"role": "system",
             "content": system_content},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        try:

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean_code = self.extract_clean_code(decoded)

            if clean_code and clean_code.startswith('def'):
                return clean_code
            else:
                raise ValueError("无法提取有效代码")

        except Exception as e:
            logger.error(f"翻译错误: {str(e)}")
            return None


def process_file(input_path, output_path):
    """处理文件并确保干净输出"""
    translator = CleanTranslator()
    if not translator.initialize():
        return False

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)

            with open(output_path, 'w', encoding='utf-8') as out_f:
                progress = tqdm(total=total_lines, desc="处理进度")

                for line in f:
                    line = line.strip()
                    if not line:
                        progress.update(1)
                        continue

                    # 解析索引和代码
                    if ':' in line:
                        idx, code = line.split(':', 1)
                        idx, code = idx.strip(), code.strip()
                    else:
                        idx, code = "Unknown", line

                    try:
                        result = translator.translate(code)
                        if result:
                            out_f.write(f"{idx}: {result}\n")
                            logger.info(f"成功处理 #{idx}")
                        else:
                            out_f.write(f"{idx}: ❌ 错误 - 无法提取有效代码\n")
                            logger.warning(f"处理失败 #{idx}")

                    except Exception as e:
                        out_f.write(f"{idx}: ❌ 错误 - {str(e)[:50]}\n")
                        logger.error(f"处理错误 #{idx}: {str(e)}")

                    progress.update(1)

                progress.close()
        return True

    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        return False

if __name__ == "__main__":
    input_file = "racket/code_rkt.txt"
    output_file = "rkt_result/rkt_2_python_with_system_prompt.txt"

    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        exit(1)

    logger.info("开始转换流程...")
    if process_file(input_file, output_file):
        logger.info(f"转换完成! 结果保存至: {output_file}")
    else:
        logger.error("转换过程中出现严重错误")
