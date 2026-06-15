"""
Pass-2 repair for translate_back_2_lrpl.py output: scan back-translation JSONL
files, find candidates where `status` is valid/success but `back_translation`
is missing or empty, and re-generate them with a slightly more permissive
prompt / extractor.

Defaults are wired to consume the output of `translate_back_2_lrpl.py`
verbatim. Single-model by design (Qwen is the strongest LRPL generator in our
setting; override with --model if needed).
"""

import logging
import os
import json
import re
import sys
from vllm import LLM, SamplingParams

# ================= Defaults (overridable via CLI) =================
DEFAULT_MODEL_PATH      = "./models/Qwen2.5-Coder-14B-Instruct"
DEFAULT_LANGS           = ['julia', 'lua', 'ocaml', 'racket', 'r']
DEFAULT_GPU_UTILIZATION = 0.95
DEFAULT_MAX_MODEL_LEN   = 4096
DEFAULT_TP_SIZE         = 2
DEFAULT_DATA_ROOT       = "./data/LowData"
# Matches translate_back_2_lrpl.py default output_template
DEFAULT_INPUT_TEMPLATE  = "{data_root}/{lang}/trans_{base}/{lang}_python_multi_trans_vllm_back.jsonl"
# ===================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _base_subdir(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))


def enhanced_extract_code(generated_text):
    """
    增强版代码提取：
    1. 支持未闭合的 Markdown 块 (处理被 max_tokens 截断的情况)
    2. 支持更多语言的关键词 (let, define, etc.)
    """
    generated_text = generated_text.strip()

    # 1. 优先尝试 Markdown 提取
    # 修改点：(?:```|$) 允许以字符串结尾作为结束，兼容未闭合的代码块
    match = re.search(r'```(?:\w+)?\s*(.*?)(?:```|$)', generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
        if code: return code  # 只有非空才返回

    # 2. 尝试关键词匹配 (裸代码)
    # 修改点：增加了 let (OCaml), define (Racket), local (Lua), import/require
    patterns = [
        r'((?:def|class|function|let|define|local)\s+\w+.*)',  # 通用定义
        r'(library\s*\(.*)',  # R library
        r'(#include\s*<.*)',  # C/C++
        r'(import\s+.*)',  # Python/Java import
    ]

    for pat in patterns:
        match = re.search(pat, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 3. Fallback: 简单的行过滤
    lines = [line for line in generated_text.split('\n')
             if line.strip() and not line.strip().lower().startswith(
            ('code:', 'python code:', 'ocaml code:', 'here is', 'julia code:', '//', '#', 'note:')
        )]

    return '\n'.join(lines).strip()


def format_prompt(python_code, target_lang):
    return f"""You are an expert code translator. Translate the following Python code to {target_lang}.

Requirements:
1. Preserve the original logic and functionality
2. Return ONLY executable {target_lang} code without explanations

Python Code:
{python_code}

{target_lang} Code:"""


def repair_empty_translations(model_path, langs, data_root, input_template,
                              gpu_util, max_model_len, tp_size):
    logger.info(f"Initializing vLLM for repair pass (Model: {model_path})...")
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
        )
        # Slightly larger max_tokens to reduce truncation in the repair pass
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=1024,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
    except Exception as e:
        logger.error(f"vLLM Init Failed: {e}")
        return

    base = _base_subdir(model_path)
    for lang in langs:
        file_path = input_template.format(data_root=data_root, lang=lang, base=base)

        if not os.path.exists(file_path):
            logger.warning(f"[{lang}] 文件不存在: {file_path}")
            continue

        logger.info(f"[{lang}] 正在检查空洞: {file_path}")

        data_items = []
        repair_tasks = []  # (d_idx, c_idx, prompt)

        # 读取并扫描空洞
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    data_items.append(item)

                    # 扫描 candidates
                    d_idx = len(data_items) - 1
                    for c_idx, cand in enumerate(item.get('translations', [])):
                        status = cand.get('status', '')
                        bt = cand.get('back_translation', None)

                        # 条件：Valid 且 (没有回译字段 或 回译是空的)
                        if status in ['valid', 'repaired_success', 'success']:
                            if bt is None or bt.strip() == "":
                                # 这是一个空洞，需要修复
                                prompt = format_prompt(cand['code'], lang)
                                repair_tasks.append((d_idx, c_idx, prompt))

        if not repair_tasks:
            logger.info(f"[{lang}] 完美！没有发现空的回译字段。")
            continue

        logger.info(f"[{lang}] 发现 {len(repair_tasks)} 个空回译，开始修复...")

        # 批量生成
        prompts = [t[2] for t in repair_tasks]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        # 填回数据
        fixed_count = 0
        for i, output in enumerate(outputs):
            d_idx, c_idx, _ = repair_tasks[i]
            generated_text = output.outputs[0].text

            # 使用增强版提取
            clean_code = enhanced_extract_code(generated_text)

            # 如果提取出来还是空的，保留原始生成文本方便 debug，或者填个标记
            if not clean_code.strip():
                logger.warning(
                    f"[{lang}] ID {data_items[d_idx].get('index')} 修复后依然为空! Raw: {generated_text[:50]}...")
                # 最后的保底：直接用原始文本
                clean_code = generated_text.strip()

            data_items[d_idx]['translations'][c_idx]['back_translation'] = clean_code
            fixed_count += 1

        # 修改输出文件名，加个 _fixed 后缀
        output_file_path = file_path.replace(".jsonl", "_fixed.jsonl")

        logger.info(f"[{lang}] 修复完成 {fixed_count} 条，正在保存至新文件: {output_file_path}")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            for item in data_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"[{lang}] 保存成功。请检查 {output_file_path} 后手动替换原文件。")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Repair pass for translate_back_2_lrpl.py: fill in empty `back_translation` "
                    "fields by re-prompting the same model with a more permissive extractor."
    )
    parser.add_argument("--model",     default=DEFAULT_MODEL_PATH)
    parser.add_argument("--langs",     nargs="+", default=DEFAULT_LANGS)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input_template", default=DEFAULT_INPUT_TEMPLATE,
                        help="Template with {data_root}/{lang}/{base} placeholders.")
    parser.add_argument("--gpu_util",      type=float, default=DEFAULT_GPU_UTILIZATION)
    parser.add_argument("--max_model_len", type=int,   default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tp_size",       type=int,   default=DEFAULT_TP_SIZE)
    args = parser.parse_args()

    repair_empty_translations(
        model_path=args.model,
        langs=args.langs,
        data_root=args.data_root,
        input_template=args.input_template,
        gpu_util=args.gpu_util,
        max_model_len=args.max_model_len,
        tp_size=args.tp_size,
    )