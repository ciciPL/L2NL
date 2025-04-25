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
    def __init__(self, model_name="../../../model/deepseek-coder-1.3b-instruct", batch_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.batch_size = batch_size  # 新增批处理大小参数

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
                device_map={"": 0},
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).eval()

            logger.info(f"模型加载成功: {self.device} | 批处理大小: {self.batch_size}")
            return True
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            return False

    def extract_clean_code(self, generated_text):
        """严格提取纯代码（原逻辑不变）"""
        pattern1 = r'(def\s[\s\S]*?)(?=\n\s*\d+:|\Z)'
        pattern2 = r'(?<=genPython:\n)([\s\S]*?)(?=\n\s*(?:Note:|Racket:|$))'

        for pattern in [pattern1, pattern2]:
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
                code = re.sub(r'Note:.*$', '', code, flags=re.MULTILINE)
                lines = [line.rstrip() for line in code.split('\n') if line.strip()]
                return '\n'.join(lines)
        return None

    def translate_batch(self, julia_code):
        """新增批量翻译方法"""
        system_content = """Convert julia to Python. Output ONLY the Python code with proper indentation."""

        # 生成批量prompt
        messages = [
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"RACKET_CODE: {code}"}
            ] for code in julia_code
        ]

        # 批量处理模板
        texts = [self.tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # 批量编码
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # 批量生成
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 批量解码
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self.extract_clean_code(d) for d in decoded]


def process_file(input_path, output_path, batch_size):
    """处理文件（修改为分批处理）"""
    translator = CleanTranslator(batch_size=batch_size)  # 可调整批大小
    if not translator.initialize():
        return False

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            total_lines = len(lines)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            progress = tqdm(total=total_lines, desc="处理进度")

            # 分批处理
            for i in range(0, total_lines, translator.batch_size):
                batch = lines[i:i + translator.batch_size]

                # 解析索引和代码
                batch_data = []
                for line in batch:
                    if ':' in line:
                        idx, code = line.split(':', 1)
                        batch_data.append((idx.strip(), code.strip()))
                    else:
                        batch_data.append(("Unknown", line))

                # 提取纯代码部分
                codes = [item[1] for item in batch_data]

                try:
                    results = translator.translate_batch(codes)
                    for (idx, _), result in zip(batch_data, results):
                        out_f.write(f"{idx}: {result if result else '# 转换失败'}\n")
                        progress.update(1)

                except Exception as e:
                    for idx, _ in batch_data:
                        out_f.write(f"{idx}: ❌ 错误 - {str(e)[:50]}\n")
                        progress.update(1)

            progress.close()
        return True

    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        return False


if __name__ == "__main__":
    input_file = "../../../dataset/julia/code.txt"
    output_file = "../julia_result/julia_2_python.txt"

    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        exit(1)

    logger.info("开始转换流程...")
    if process_file(input_file, output_file,32):
        logger.info(f"转换完成! 结果保存至: {output_file}")
    else:
        logger.error("转换过程中出现严重错误")
