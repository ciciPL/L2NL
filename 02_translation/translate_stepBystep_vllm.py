import logging
import os
import json
import re
import ast
import sys
import gc
from os.path import basename

import torch
from transformers import AutoTokenizer  # 仅用于apply_chat_template
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

# 生成配置（CLI 可覆盖）
TEMP_SCHEDULE = [
    (0.0, 1),
    (0.7, 3),
    (0.9, 3),
    (1.1, 3)
]

MAX_REPAIR_ATTEMPTS = 3
MAX_DATA_NUM = None        # 默认全跑；CLI 可设上限
GPU_UTILIZATION = 0.90
TENSOR_PARALLEL = 2        # 根据 GPU 数调整
MAX_MODEL_LEN = 4096
# ===============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("translation_vllm_forward.log"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ================= 辅助函数 =================

def validate_syntax(code):
    if not code: return False, "Empty code", "Empty"
    try:
        ast.parse(code)
        return True, None, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}", "SyntaxError"
    except Exception as e:
        return False, str(e), type(e).__name__


def extract_clean_code(generated_text):
    # 1. XML 优先
    xml_match = re.search(r'<PYTHON>(.*?)</PYTHON>', generated_text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        content = xml_match.group(1).strip()
        md_match = re.search(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL | re.IGNORECASE)
        if md_match: return md_match.group(1).strip()
        return content

    # 2. Markdown Fallback
    md_match = re.search(r'```python\s*(.*?)\s*```', generated_text, re.DOTALL | re.IGNORECASE)
    if md_match: return md_match.group(1).strip()

    # 3. 暴力清洗 (移除 Note 等) - 关键修改：不再截断 def
    clean_text = generated_text
    garbage_patterns = [r'\nNote:', r'\nExplanation:', r'The Python code']
    for pattern in garbage_patterns:
        parts = re.split(pattern, clean_text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) > 1: clean_text = parts[0]

    return clean_text.strip()

def load_data(file_path, max_num=None):
    data = []
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if max_num and len(data) >= max_num: break
        line = line.strip()
        if not line: continue
        if ':' in line:
            idx, code = line.split(':', 1)
            data.append({"index": idx.strip(), "code": code.strip()})
    return data

def load_json_data(file_path, max_num=None):
    data = []
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_num and len(data) >= max_num: break
            js = json.loads(line)
            if not line: continue
            code = js['code']
            i=i+1
            data.append({"index": str(i), "code": code.strip()})
    return data


def format_prompt_translate(source_code, lang):
    return f"""You are an expert code translator. Translate the following {lang} code to Python.

Requirements:
1. Preserve the original logic and functionality
2. Use Python idioms and best practices
3. CRITICAL: Output ONLY the Python code logic wrapped inside <PYTHON> and </PYTHON> tags.
4. Do NOT use markdown code blocks (like ```python).
5. Do NOT output explanations or notes or testcase.

Example:

Input:
func add(a, b) {{ return a + b }}

Output:
<PYTHON>
def add(a, b):
    return a + b
</PYTHON>

Real Task:

{lang} Input:
{source_code}

Output:"""


def format_prompt_repair(source_code, failed_candidate, error_log, lang):
    return f"""You are an expert code debugger.
Source ({lang}):
{source_code}

Your previous Python translation:
{failed_candidate}

Error Message:
{error_log}

Fix the Python code based on the error. Return ONLY the fixed code.
Python Code:"""


# ================= 核心处理逻辑 =================

def process_single_language(llm, tokenizer, lang, input_file, output_file,jsonFLag=False):
    logger.info("=" * 60)
    logger.info(f"开始处理语言: {lang}")
    logger.info("=" * 60)
    if jsonFLag:
        raw_data = load_json_data(input_file, max_num=MAX_DATA_NUM)
    else:
        raw_data = load_data(input_file, max_num=MAX_DATA_NUM)
    if not raw_data: return

    # 初始化结果结构
    # 使用字典方便索引: all_results[index]
    all_results = {str(item['index']): {"index": item['index'], "source_code": item['code'], "translations": []}
                   for item in raw_data}

    # ---------------------------------------------------------
    # Phase 1: 多温度 Batch 生成 (并行化)
    # ---------------------------------------------------------
    logger.info(f"[{lang}] Phase 1: 多温度生成启动")

    # 我们需要针对每个 temperature 跑一轮，因为 vLLM 的 generate 接受统一的 sampling_params
    # 或者可以将所有请求构建好，按 temperature 分组发送

    logger.info(f"[{lang}] Phase 1: 多温度生成启动")

    for temp, n_samples in TEMP_SCHEDULE:
        logger.info(f"[{lang}] Generating with Temp={temp}, Samples={n_samples}")

        prompts = []
        meta_data = []

        for item in raw_data:
            # === [修改点 1]：不再直接使用纯文本，而是构建 Messages ===
            system_prompt = "You are an expert code translator."
            user_content = format_prompt_translate(item['code'], lang)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            full_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            prompts.append(full_prompt)
            meta_data.append(item['index'])

        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=512,  # 建议设大一点，防止截断
            n=n_samples,
            # === [修改点 3]：确保 Stop Token 包含模型特定的结束符 ===
            # DeepSeek 通常使用 <|EOT|> 或 <|im_end|>，transformers 的 chat_template 会处理好，
            # 但在 vLLM 中显式加上这些会更安全。
            stop=["<|endoftext|>", "<|im_end|>", "<|EOT|>"]
        )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        # 收集结果
        for i, output in enumerate(outputs):
            idx = meta_data[i]
            # output.outputs 包含 n 个生成结果
            for sample_out in output.outputs:
                generated_text = sample_out.text
                clean_code = extract_clean_code(generated_text)

                all_results[idx]["translations"].append({
                    "code": clean_code,
                    "temperature": temp,
                    "raw_output": generated_text,
                    # 状态稍后统一验证
                })

    # ---------------------------------------------------------
    # Phase 2: 验证语法
    # ---------------------------------------------------------
    logger.info(f"[{lang}] Phase 2: 验证语法")
    failed_indices = []

    for idx, data in all_results.items():
        candidates = data["translations"]
        valid_count = 0
        for cand in candidates:
            is_valid, err_msg, err_type = validate_syntax(cand['code'])
            cand['status'] = 'valid' if is_valid else 'invalid'
            cand['error'] = err_msg
            if is_valid: valid_count += 1

        data['valid_count'] = valid_count
        if valid_count == 0:
            data['status'] = 'all_failed'
            failed_indices.append(idx)
        else:
            data['status'] = 'success'

    # 保存中间结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(all_results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(json.dumps(all_results[idx], ensure_ascii=False) + '\n')

    if not failed_indices:
        return

    # ---------------------------------------------------------
    # Phase 3: 错误修复 (Batch vLLM)
    # ---------------------------------------------------------
    logger.info(f"[{lang}] Phase 3: 开始修复 {len(failed_indices)} 个样本 (Batch Mode)")

    # 修复也是一轮一轮来，最大 MAX_REPAIR_ATTEMPTS 轮
    current_failed_indices = failed_indices[:]

    sampling_params_repair = SamplingParams(
        temperature=0.2,  # 修复通常用低温
        max_tokens=512,
        n=1
    )

    for attempt in range(MAX_REPAIR_ATTEMPTS):
        if not current_failed_indices: break

        logger.info(
            f"[{lang}] Repair Attempt {attempt + 1}/{MAX_REPAIR_ATTEMPTS}, Count: {len(current_failed_indices)}")

        prompts = []
        batch_indices = []  # 记录这一轮修的是哪些

        # 准备数据
        for idx in current_failed_indices:
            item = all_results[idx]
            # 找最短的 invalid 代码作为基底 (逻辑同 Script A)
            invalids = [c for c in item['translations'] if c['status'] == 'invalid']
            # 注意：如果是第2次修复，应该基于上一次修复失败的代码，
            # 但为了简化逻辑，我们这里总是取当前池子里最短的 invalid 代码。
            # 如果你想严格遵循"基于上一次修复结果"，需要维护一个 current_code_state。
            # 这里为了性能，我们假设池子里只要是 invalid 都可以用来参考。
            # 更严谨的做法是：每次修复产生的新 invalid 代码加入池子，下次取最新的。

            # 简单策略：取所有 invalid 中最短的一个
            invalids.sort(key=lambda x: len(x['code']) if x['code'] else 9999)
            best_fail = invalids[0]

            # 构造 Chat Prompt (Repair 需要对话格式)
            prompt_str = format_prompt_repair(item['source_code'], best_fail['code'], best_fail['error'], lang)

            # 使用 tokenizer 应用 chat 模板，因为 vLLM 输入是 text
            messages = [{"role": "user", "content": prompt_str}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            prompts.append(full_prompt)
            batch_indices.append(idx)

        # 批量生成
        outputs = llm.generate(prompts, sampling_params_repair, use_tqdm=True)

        # 处理结果 & 更新状态
        next_round_failed = []
        success_count_this_round = 0

        for i, output in enumerate(outputs):
            idx = batch_indices[i]
            generated_text = output.outputs[0].text
            new_code = extract_clean_code(generated_text)

            valid, err, _ = validate_syntax(new_code)

            # 将新结果加入 translations
            new_entry = {
                "code": new_code,
                "temperature": 0.2,
                "status": "valid" if valid else "invalid",
                "repaired": True,
                "attempt": attempt + 1,
                "error": err if not valid else None
            }
            all_results[idx]['translations'].append(new_entry)

            if valid:
                all_results[idx]['status'] = 'repaired_success'
                success_count_this_round += 1
            else:
                next_round_failed.append(idx)

        current_failed_indices = next_round_failed
        logger.info(f"[{lang}] Round {attempt + 1} repaired: {success_count_this_round}")

    # ---------------------------------------------------------
    # Phase 4: 最终保存
    # ---------------------------------------------------------
    logger.info(f"[{lang}] 处理完成。")
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx in sorted(all_results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            f.write(json.dumps(all_results[idx], ensure_ascii=False) + '\n')


def _base_subdir(model_path: str) -> str:
    """Map model path → server-side experimental folder name."""
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if '6.7b'  in model_path: return 'ds_mid3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return basename(model_path.rstrip('/'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-temperature pivot-translation (LRPL → Python) with AST-based repair, via vLLM."
    )
    parser.add_argument("--models", nargs="+",
                        default=[
                            "./models/deepseek-coder-1.3b-instruct",
                            "./models/deepseek-coder-6.7b-instruct",
                            "./models/Llama-3.1-8B-Instruct",
                            "./models/Seed-Coder-8B-Instruct",
                            "./models/Qwen2.5-Coder-14B-Instruct",
                        ],
                        help="One or more HuggingFace model paths.")
    parser.add_argument("--langs", nargs="+",
                        default=['julia', 'lua', 'ocaml', 'racket', 'r'],
                        help="Languages to process. Use ['ruby'] for CSN run.")
    parser.add_argument("--data_root", default="./data/LowData",
                        help="Root directory holding `<lang>/code_<lang>.txt` (LRPL) "
                             "or the CSN parent. Default: ./data/LowData")
    parser.add_argument("--input_template",
                        default="{data_root}/{lang}/code_{lang}.txt",
                        help="Input file template; can reference {data_root} and {lang}.")
    parser.add_argument("--output_template",
                        default="{data_root}/{lang}/trans_{base}/{lang}_python_multi_trans_vllm.jsonl",
                        help="Output file template; can reference {data_root}, {lang}, {base}.")
    parser.add_argument("--json_input", action="store_true",
                        help="Read input as JSONL with a 'code' field instead of `idx:code` lines "
                             "(set this for the CSN Ruby file).")
    parser.add_argument("--max_data", type=int, default=None,
                        help="Cap samples per language (default: all).")
    parser.add_argument("--gpu_util",   type=float, default=GPU_UTILIZATION)
    parser.add_argument("--tp_size",    type=int,   default=TENSOR_PARALLEL)
    parser.add_argument("--max_model_len", type=int, default=MAX_MODEL_LEN)
    args = parser.parse_args()

    # MAX_DATA_NUM lives at module scope; process_single_language() reads it
    # via load_data(... max_num=MAX_DATA_NUM). Rebind it here so the CLI flag wins.
    MAX_DATA_NUM = args.max_data

    for MODEL_PATH in args.models:
        base_path = _base_subdir(MODEL_PATH)
        logger.info(f"Initializing vLLM for {MODEL_PATH} ...")
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_util,
            tensor_parallel_size=args.tp_size,
            max_model_len=args.max_model_len,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        for lang in args.langs:
            input_path  = args.input_template.format(data_root=args.data_root, lang=lang, base=base_path)
            output_path = args.output_template.format(data_root=args.data_root, lang=lang, base=base_path)
            try:
                process_single_language(llm, tokenizer, lang, input_path, output_path,
                                        jsonFLag=args.json_input)
            except Exception as e:
                logger.error(f"Error processing {lang}: {e}")
                import traceback
                traceback.print_exc()

        # ---- GPU memory cleanup between models ----
        logger.info(f"Unloading model: {MODEL_PATH} ...")
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        destroy_model_parallel()
        logger.info("GPU memory released. Ready for next model.")
        logger.info("-" * 60)

    logger.info("All Done.")
