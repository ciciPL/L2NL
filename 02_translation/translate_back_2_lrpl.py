import gc
import logging
import os
import json
import re
import sys

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vllm_back_translation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ================== 核心逻辑函数 (无需修改) ==================
def extract_clean_code(generated_text):
    """
    回译提取器：优先提取 <CODE>...</CODE>
    """
    # 1. XML 优先 (不区分大小写)
    xml_match = re.search(r'<CODE>(.*?)</CODE>', generated_text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        content = xml_match.group(1).strip()
        # 清洗可能存在的 Markdown
        md_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if md_match:
            return md_match.group(1).strip()
        return content

    # 2. Fallback: Markdown
    md_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', generated_text, re.DOTALL | re.IGNORECASE)
    if md_match: return md_match.group(1).strip()

    # 3. Fallback: 暴力清洗 (针对回译通常不用那么复杂，只要去掉 Note 即可)
    clean_text = generated_text
    garbage_patterns = [r'\nNote:', r'\nExplanation:', r'The code:']
    for pattern in garbage_patterns:
        parts = re.split(pattern, clean_text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1: clean_text = parts[0]

    return clean_text.strip()
# def extract_clean_code(generated_text):
#     """
#     增强版代码提取：
#     1. 支持未闭合的 Markdown 块 (处理被 max_tokens 截断的情况)
#     2. 支持更多语言的关键词 (let, define, etc.)
#     """
#     generated_text = generated_text.strip()
#
#     # 1. 优先尝试 Markdown 提取
#     # 修改点：(?:```|$) 允许以字符串结尾作为结束，兼容未闭合的代码块
#     match = re.search(r'```(?:\w+)?\s*(.*?)(?:```|$)', generated_text, re.DOTALL | re.IGNORECASE)
#     if match:
#         code = match.group(1).strip()
#         if code: return code  # 只有非空才返回
#
#     # 2. 尝试关键词匹配 (裸代码)
#     # 修改点：增加了 let (OCaml), define (Racket), local (Lua), import/require
#     patterns = [
#         r'((?:def|class|function|let|define|local)\s+\w+.*)',  # 通用定义
#         r'(library\s*\(.*)',  # R library
#         r'(#include\s*<.*)',  # C/C++
#         r'(import\s+.*)',  # Python/Java import
#     ]
#
#     for pat in patterns:
#         match = re.search(pat, generated_text, re.DOTALL)
#         if match:
#             return match.group(1).strip()
#
#     # 3. Fallback: 简单的行过滤
#     lines = [line for line in generated_text.split('\n')
#              if line.strip() and not line.strip().lower().startswith(
#             ('code:', 'python code:', 'ocaml code:', 'here is', 'julia code:', '//', '#', 'note:')
#         )]
#
#     return '\n'.join(lines).strip()


def format_prompt(python_code, target_lang):
    ONE_SHOT_EXAMPLES = {
        "julia": """
    Input:
    def add(a, b):
        return a + b

    Output:
    <CODE>
    function add(a, b)
        return a + b
    end
    </CODE>
    """,
        "lua": """
    Input:
    def add(a, b):
        return a + b

    Output:
    <CODE>
    function add(a, b)
        return a + b
    end
    </CODE>
    """,
        "ocaml": """
    Input:
    def add(a, b):
        return a + b

    Output:
    <CODE>
    let add a b = a + b
    </CODE>
    """,
        "racket": """
    Input:
    def add(a, b):
        return a + b

    Output:
    <CODE>
    (define (add a b)
      (+ a b))
    </CODE>
    """,
        "r": """
    Input:
    def add(a, b):
        return a + b

    Output:
    <CODE>
    add <- function(a, b) {
      return(a + b)
    }
    </CODE>
    """
    }
    example_str = ONE_SHOT_EXAMPLES.get(target_lang, "")
    """构造回译 Prompt"""
    return f"""You are an expert code translator. Translate the following Python code to {target_lang}.

Requirements:
1. Preserve the original logic and functionality.
2. Use {target_lang} idioms and best practices.
3. CRITICAL: Output ONLY the {target_lang} code logic wrapped inside <CODE> and </CODE> tags.
4. Do NOT use markdown code blocks.
5. Do NOT output explanations or notes or testcase.

Example:
{example_str}

Real Task:

Input:
{python_code}

Output:"""


def process_single_file(llm, tokenizer, sampling_params, lang, input_file, output_file):
    """处理单个文件的回译任务 (Python -> Source Lang)"""
    if not os.path.exists(input_file):
        logger.warning(f"[{lang}] 文件不存在，跳过: {input_file}")
        return

    logger.info(f"[{lang}] 读取文件: {input_file}")

    # 1. 读取数据
    data_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_items.append(json.loads(line))

    # 2. 收集任务 (Flatten)
    prompts = []
    task_indices = []  # 记录 (data_idx, cand_idx)

    for d_idx, item in enumerate(data_items):
        translations = item.get('translations', [])
        for c_idx, cand in enumerate(translations):
            # 只回译 status 正常的代码
            # 注意：有些失败的代码可能也想回译看看（取决于你的实验设计），但通常只回译能跑通的
            status = cand.get('status', '')
            temperature = cand.get('temperature', '')

            # 这里的条件按你原本的逻辑保持不变
            if status in ['valid', 'repaired_success', 'success'] or temperature == 0.0:
                # === [修改核心开始] ===

                # 1. 获取纯文本指令 (假设 format_prompt 返回的是 "Translate Python to Lua...")
                user_content = format_prompt(cand['code'], lang)

                # 2. 构造对话结构
                messages = [
                    {"role": "system",
                     "content": f"You are an expert code translator. Translate the Python code to {lang}."},
                    {"role": "user", "content": user_content}
                ]

                # 3. 应用 Chat Template (解决 DeepSeek 发疯写 Note 的问题)
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                prompts.append(full_prompt)
                task_indices.append((d_idx, c_idx))

                # === [修改核心结束] ===

    if not prompts:
        logger.info(f"[{lang}] 没有需要回译的候选代码。")
        return

    logger.info(f"[{lang}] 收集到 {len(prompts)} 条回译任务，开始 vLLM 生成...")

    # 3. vLLM 批量推理
    # 确保 sampling_params 里包含了 stop=["<|im_end|>", "<|eot_id|>", ...]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    # 4. 结果回填
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        # 注意：这里也建议检查 extract_clean_code 是否能处理目标语言
        # 如果目标语言是 Racket (Lisp风格)，Markdown 提取逻辑可能通用，
        # 但如果是纯正则，需确保兼容目标语言语法。
        clean_code = extract_clean_code(generated_text)

        d_idx, c_idx = task_indices[i]
        data_items[d_idx]['translations'][c_idx]['back_translation'] = clean_code

    # 5. 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"[{lang}] 结果已保存: {output_file}")


# ================== Main Control ==================

def _base_subdir(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Back-translate Python pivot code → LRPL via vLLM (AST-validation prep)."
    )
    parser.add_argument("--models", nargs="+",
                        default=[
                            "./models/deepseek-coder-1.3b-instruct",
                            "./models/deepseek-coder-6.7b-instruct",
                            "./models/Llama-3.1-8B-Instruct",
                            "./models/Seed-Coder-8B-Instruct",
                            "./models/Qwen2.5-Coder-14B-Instruct",
                        ])
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'racket', 'r'],
                        help="Target LRPLs for back-translation. Use ['ruby'] for CSN run.")
    parser.add_argument("--data_root", default="./data/LowData",
                        help="Root directory (default: ./data/LowData; use ./data/CSN for CSN run).")
    parser.add_argument("--input_template",
                        default="{data_root}/{lang}/trans_{base}/{lang}_python_multi_trans_vllm.jsonl")
    parser.add_argument("--output_template",
                        default="{data_root}/{lang}/trans_{base}/{lang}_python_multi_trans_vllm_back.jsonl")
    parser.add_argument("--gpu_util", type=float, default=0.95)
    parser.add_argument("--tp_size",  type=int,   default=2)
    parser.add_argument("--max_model_len", type=int, default=4096)
    args = parser.parse_args()

    for MODEL_PATH in args.models:
        base_path = _base_subdir(MODEL_PATH)
        logger.info(f"Initializing vLLM (Model: {MODEL_PATH})")
        try:
            llm = LLM(
                model=MODEL_PATH,
                trust_remote_code=True,
                gpu_memory_utilization=args.gpu_util,
                max_model_len=args.max_model_len,
                tensor_parallel_size=args.tp_size,
            )
            # Greedy decoding (back-translation doesn't need diversity)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=512,
                stop=["<|endoftext|>", "<|im_end|>"],
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        except Exception as e:
            logger.error(f"vLLM init failed: {e}")
            sys.exit(1)

        for lang in args.langs:
            input_path  = args.input_template.format(data_root=args.data_root, lang=lang, base=base_path)
            output_path = args.output_template.format(data_root=args.data_root, lang=lang, base=base_path)
            try:
                process_single_file(llm, tokenizer, sampling_params, lang, input_path, output_path)
            except Exception as e:
                logger.error(f"[{lang}] processing failed: {e}")

        logger.info(f"Unloading model: {MODEL_PATH} ...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()
        logger.info("GPU memory released.")
        logger.info("-" * 60)

    logger.info("All Done.")