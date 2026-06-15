import gc
import json
import os
import sys
import torch
import logging
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# Defaults (CLI overrides in __main__)
# ============================================================
MAX_DATA_NUM = None       # default: all; override with --max_data
USE_FEW_SHOT = True

DEFAULT_MODELS = [
    "./models/deepseek-coder-1.3b-instruct",
    "./models/deepseek-coder-6.7b-instruct",
    "./models/Llama-3.1-8B-Instruct",
    "./models/Seed-Coder-8B-Instruct",
    "./models/Qwen2.5-Coder-14B-Instruct",
]
DEFAULT_LANGS     = ['julia', 'lua', 'ocaml', 'r', 'racket']
DEFAULT_DATA_ROOT = "./data/LowData"

# ============================================================
# Few-Shot Data (内容保持不变，格式将在构建 Prompt 时动态适配)
# ============================================================
FEW_SHOTS_DATA = {
    "ocaml": [
        {
            "code": "def make_html_safe(s: str) -> str:\n    return s",
            "trace": ["return s"],
            "summary": "Replace any input in string s to enhance safe."
        },
        {
            "code": "def trailing_zeroes(num: int) -> int:\n    if num == 0:\n        return 32\n    p = 0\n    while (num >> p) & 1 == 0:\n        p += 1\n    return p",
            "trace": ["while (num >> p) & 1 == 0", "p += 1"],
            "summary": "Counts trailing zero bits in hashed value using bitwise operations."
        },
        {
            "code": "def hamming(first: str, second: str) -> int:\n    if len(first) != len(second):\n        return -1\n    return sum(1 for x, y in zip(first, second) if x != y)",
            "trace": ["if len(first) != len(second)", "sum(1 for x, y in zip(first, second) if x != y)"],
            "summary": "Calculates Hamming distance between two strings of equal length."
        }
    ],
    "r": [
        {
            "code": "def PowerStack_Calc(Power, N):\n    result = N * Power\n    return result",
            "trace": ["result = N * Power"],
            "summary": "Calculates power stack by multiplying single cell power by number of cells."
        },
        {
            "code": "def trackOpposite(f_tr):\n    l_d = f_tr + 180.0\n    while l_d >= 360.0:\n        l_d = l_d - 360.0\n    return l_d",
            "trace": ["l_d = f_tr + 180.0", "while l_d >= 360.0: l_d = l_d - 360.0"],
            "summary": "Function to calculate the opposite direction by adding 180 degrees and normalizing to 0-360 range."
        },
        {
            "code": "import re\ndef normalize_meaning(source_string):\n    source_string = source_string.replace('\"', '&quot;')\n    source_string = source_string.replace(\"'\", '&apos;')\n    source_string = source_string.replace('\\n', ' ')\n    source_string = source_string.replace('\\t', ' ')\n    source_string = re.sub(r'<.*?>', '', source_string)\n    source_string = re.sub(r' {2,}', ' ', source_string)\n    return source_string.strip()",
            "trace": ["source_string.replace('\"', '&quot;')", "re.sub('<.*?>', '', source_string)",
                      "source_string.strip()"],
            "summary": "Normalizes a string by escaping HTML tags, replacing quotes, removing newlines, tabs, and extra spaces, and trimming whitespace."
        }
    ],
    "julia": [
        {
            "code": "import os\ndef get_id_from_repo(repo_dir):\n    # this is the top of the repo directory\n    # return repo_dir.split('/')[-1]\n    # this is the id of the file in the repo\n    return os.path.basename(repo_dir)",
            "trace": ["return os.path.basename(repo_dir)"],
            "summary": "Extracts Google Drive File/Folder ID from repository directory path."
        },
        {
            "code": "def merge_segdb(segdbs):\n    # Concatenates a list of lists\n    return [item for sublist in segdbs for item in sublist]",
            "trace": ["return [item for sublist in segdbs for item in sublist]"],
            "summary": "Concatenates a list of segdbs into a single segdb."
        },
        {
            "code": "def linear_cell_to_tuple(c1, repeat_units):\n    # c1 = int(c1)\n    c1x = c1 % repeat_units[0]\n    c1y = ((c1 - c1x) // repeat_units[0]) % repeat_units[1]\n    c1z = (c1 - c1x - c1y * repeat_units[0]) // (repeat_units[0] * repeat_units[1])\n    return c1x, c1y, c1z",
            "trace": ["c1x = c1 % repeat_units[0]", "c1y = ((c1 - c1x) // repeat_units[0]) % repeat_units[1]"],
            "summary": "Converts a linear index to a 3D tuple using lexicographic indexing."
        }
    ],
    "lua": [
        {
            "code": "def _translate_message(message):\n    if message:\n        return {\n            'id': message.id,\n            'project_id': message.project_id,\n            'request_id': message.request_id,\n            'resource_type': message.resource_type,\n            'resource_uuid': message.resource_uuid,\n            'event_id': message.event_id,\n            'message_level': message.message_level,\n            'created_at': message.created_at,\n            'expires_at': message.expires_at,\n        }\n    else:\n        return None",
            "trace": ["def _translate_message(message)", "return { \"id\": message.id, ... }"],
            "summary": "Translates Message model to a dictionary."
        },
        {
            "code": "def compare_rule_hits_count(r1, r2, diff):\n    if r1 is None or r2 is None:\n        return False\n    r1_count = 0\n    r2_count = 0\n    if r1.get('report') and r1['report'].get('meta') and r1['report']['meta'].get('count') is not None:\n        r1_count = r1['report']['meta']['count']\n    if r2.get('report') and r2['report'].get('meta') and r2['report']['meta'].get('count') is not None:\n        r2_count = r2['report']['meta']['count']\n    diff['hits1'] = r1_count\n    diff['hits2'] = r2_count\n    return r1_count == r2_count",
            "trace": ["r1_count = r1.get(\"report\").get(\"meta\").get(\"count\")", "return r1_count == r2_count"],
            "summary": "Compares rule hit counts and updates diff structure."
        },
        {
            "code": "def getBoundsOverlap(bb1, bb2):\n    x1, y1, x2, y2 = bb1[0], bb1[1], bb1[2], bb1[3]\n    x3, y3, x4, y4 = bb2[0], bb2[1], bb2[2], bb2[3]\n    minX = max(x1, x3)\n    minY = max(y1, y3)\n    maxX = min(x2, x4)\n    maxY = min(y2, y4)\n    if maxX < minX or maxY < minY:\n        return False\n    return [[minX, minY, maxX, maxY]]",
            "trace": ["minX = max(x1, x3); maxX = min(x2, x4)", "if maxX < minX or maxY < minY: return False"],
            "summary": "Calculates intersection of two bounding boxes."
        }
    ],
    "racket": [
        {
            "code": "import math\ndef get_color_towards(color1, color2, fraction):\n    return [math.floor(x + fraction * (y - x)) for x, y in zip(color1, color2)]",
            "trace": ["math.floor(x + fraction * (y - x))"],
            "summary": "Calculates a color between two given colors based on a fraction."
        },
        {
            "code": "def _map_boolean_to_human_readable(boolean, resource, token):\n    if boolean:\n        return \"Yes\"\n    else:\n        return \"No\"",
            "trace": ["if boolean: return \"Yes\"", "return \"No\""],
            "summary": "Maps boolean to Yes or No for human readability."
        },
        {
            "code": "def on_segment(p, q, r):\n    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and\n            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))",
            "trace": ["q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0])"],
            "summary": "Checks if point q lies on line segment pr for collinear points p, q, r."
        }
    ]
}


# ============================================================
# 数据加载与处理逻辑
# ============================================================

def load_merged_data(lang, max_num=None, data_root=None, base_tag="qwen3th"):
    """
    同时读取 代码文件 和 结构(Snippets)文件。
    增加逻辑：如果 cleaned_seqs_pred 为空，尝试读取 function_def。
    """
    root = data_root or DEFAULT_DATA_ROOT
    base_dir = f"{root}/{lang}/trans_{base_tag}"
    code_file = os.path.join(base_dir, f"{lang}_python_best_candidate_vllm_B0.5_S0.5.jsonl")
    struct_file = os.path.join(base_dir, f"python_{lang}_structure_B0.5_S0.5_preds.jsonl")

    if not os.path.exists(code_file) or not os.path.exists(struct_file):
        logger.error(f"Missing file(s) for {lang}.\nCode: {code_file}\nStruct: {struct_file}")
        return []

    merged_data = []
    current_idx = 1

    with open(code_file, 'r', encoding='utf-8') as f_c, open(struct_file, 'r', encoding='utf-8') as f_s:
        code_lines = f_c.readlines()
        struct_lines = f_s.readlines()

        limit = min(len(code_lines), len(struct_lines))
        if max_num:
            limit = min(limit, max_num)

        for i in range(limit):
            try:
                c_item = json.loads(code_lines[i])
                s_item = json.loads(struct_lines[i])

                best_code = c_item.get("best_python_code", "")

                # ========================================
                # Fallback Logic: Trace -> Function Def -> Empty
                # ========================================
                snippets = s_item.get("cleaned_seqs_pred", [])

                # 1. 如果 snippets 是空的或者全是空字符串
                if not snippets or (
                        isinstance(snippets, list) and not any(s.strip() for s in snippets if isinstance(s, str))):
                    # 2. 尝试获取 function_def
                    func_def = s_item.get("function_def", "")
                    if func_def and isinstance(func_def, str) and func_def.strip():
                        snippets = [func_def]
                    else:
                        # 3. 实在没有，就是空列表
                        snippets = []

                original_id = c_item.get("index") or c_item.get("id")

                merged_data.append({
                    "idx": current_idx,
                    "original_id": original_id,
                    "code": best_code,
                    "snippets": snippets
                })
                current_idx += 1

            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(merged_data)} items for {lang}")
    return merged_data


# ============================================================
# Prompt Construction (Hybrid Style)
# ============================================================
def clean_extracted_summary(raw_text):
    """
    从输出中清洗 <SUMMARY> 标签
    """
    # 提取 <SUMMARY> 之后的内容
    if "<SUMMARY>" in raw_text:
        content = raw_text.split("<SUMMARY>")[-1]
    else:
        content = raw_text

    # 移除结束标签
    content = content.replace("</SUMMARY>", "")

    # 有时候模型可能还是输出了 # 或注释符号，简单清洗一下
    content = content.strip()
    # 如果开头是 #，去掉它
    if content.startswith("#"):
        content = content.lstrip("#").strip()

    return content


# ============================================================
# 1. 修改格式化函数，区分 Input 和 Output
# ============================================================
def format_block_annotation_style(code, snippets = None, summary=None, return_parts=False):
    """
    格式化单个数据块。
    return_parts=True 时，返回 (user_text, assistant_text) 用于构建多轮对话
    """

    # --- 1. 格式化 Trace ---
    formatted_snippets = []
    trace_content=''
    if snippets:
        if isinstance(snippets, list):
            for snip in snippets:
                if snip and isinstance(snip, str) and snip.strip():
                    formatted_snippets.append(f"# > {snip.strip()}")
        elif isinstance(snippets, str) and snippets.strip():
            formatted_snippets.append(f"# > {snippets.strip()}")

        trace_content = "\n".join(formatted_snippets)

    # --- 2. 拼接 User 部分 (Code + Trace + Instruction) ---
    text = "# ==================================================\n"
    text += "# [USER INPUT CODE]\n"
    text += f"{code.strip()}\n\n"
    if snippets:
        text += "# [KEY LOGIC TRACE]\n"

    if trace_content:
        text += f"{trace_content}\n\n"
    else:
        text += "\n"

        # 指令放在 User 结尾
    text += "Concisely summarize the code provided in 1-3 sentences.\n"

    # --- 3. 处理 Assistant 部分 (Summary) ---
    assistant_text = ""
    if summary:
        assistant_text = f"<SUMMARY>\n{summary.strip()}\n</SUMMARY>"

    # 如果需要拆分返回 (用于多轮对话构建)
    if return_parts:
        return text, assistant_text

    # 如果只是为了打印或者旧逻辑（通常不会走到这里了）
    return text + (assistant_text + "\n" if assistant_text else "")


# ============================================================
# 2. 修改 Prompt 构建函数，生成多轮对话 List
# ============================================================
def build_hybrid_prompt(lang, code, snippets):
    """
    构建符合 Chat 模型习惯的多轮对话 Prompt
    Structure:
      System
      User (Ex1) -> Assistant (Ex1 Output)
      User (Ex2) -> Assistant (Ex2 Output)
      User (Target)
    """

    # System Message
    system_text = """You're a specialized AI assisting with Python code summaries, deeply knowledgeable in computer science.

GUIDELINES:
1. Code Analysis: Read the source code under '# [USER INPUT CODE]'.
2. Trace Usage: The '# [KEY LOGIC TRACE]' section provides execution hints. Use it as an auxiliary aid to understand the core functionality, but avoid describing low-level implementation details (e.g., specific variable manipulations) in your summary. Focus on the high-level intent.
3. Output Format: Output your natural language summary inside <SUMMARY> and </SUMMARY> tags.

Examples:"""

    messages = [{"role": "system", "content": system_text}]

    # (A) 添加 Few-Shot Examples (拆分为 User/Assistant 对)
    if USE_FEW_SHOT and lang in FEW_SHOTS_DATA:
        examples = FEW_SHOTS_DATA[lang]
        for ex in examples:
            # 获取 User 部分 和 Assistant 部分
            user_part, assistant_part = format_block_annotation_style(
                ex['code'], ex['trace'], ex['summary'], return_parts=True
            )

            # 添加一轮对话历史
            messages.append({"role": "user", "content": user_part})
            messages.append({"role": "assistant", "content": assistant_part})

    # (B) 添加 Current Query (Target)
    # 只有 User 部分，没有 Assistant 部分
    target_user_part, _ = format_block_annotation_style(
        code, snippets, summary=None, return_parts=True
    )

    messages.append({"role": "user", "content": target_user_part})

    return messages


def process_language_batch(llm, lang, output_txt_file, data_root=None, base_tag="qwen3th"):
    debug_file = output_txt_file.replace(".txt", "_debug.jsonl")

    # 加载数据
    data = load_merged_data(lang, max_num=MAX_DATA_NUM, data_root=data_root, base_tag=base_tag)
    if not data:
        return

    # 获取 Tokenizer 用于应用 Chat 模板
    tokenizer = llm.get_tokenizer()

    # 构建 Prompts
    prompts = []
    logger.info(f"Building Hybrid Chat prompts for {lang}...")

    for item in data:
        # 1. 获取结构化的 Messages
        messages = build_hybrid_prompt(lang, item['code'], item['snippets'])

        # 2. 应用 Chat Template
        # tokenize=False: 返回字符串而不是 token ids
        # add_generation_prompt=True: 自动添加 assistant 的引导头 (例如 <|start_header_id|>assistant...)
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        prompts.append(final_prompt)
        item['final_prompt'] = final_prompt

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        max_tokens=128,
        n=1,
        stop=['</SUMMARY>', '<|eot_id|>', '<|end_of_text|>']
    )

    # 执行推理
    logger.info(f"Running inference for {len(prompts)} items...")
    outputs = llm.generate(prompts, sampling_params)

    # 保存结果
    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    logger.info("💾 Saving results...")

    with open(output_txt_file, 'w', encoding='utf-8') as f_out, \
            open(debug_file, 'w', encoding='utf-8') as f_debug:

        for i, o in enumerate(outputs):
            item = data[i]
            generated_text = o.outputs[0].text

            # 清洗结果
            clean_summary = clean_extracted_summary(generated_text)
            summary_inline = clean_summary.replace('\n', ' ')

            # [关键格式] index \t summary
            f_out.write(f"{item['idx']}\t{summary_inline}\n")

            # [Debug 文件]
            debug_entry = {
                "index": item['idx'],
                "original_id": item['original_id'],
                "lang": lang,
                "prompt": item['final_prompt'],
                "raw_output": generated_text,
                "clean_summary": summary_inline
            }
            f_debug.write(json.dumps(debug_entry, ensure_ascii=False) + '\n')

    logger.info(f"✅ Results saved to {output_txt_file}")


# ============================================================
# 主入口 (Main)
# ============================================================

def _base_tag(model_path: str) -> str:
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Alternative summary generation using the ICSE-2025 chat-template prompt structure "
                    "(kept for ablation against finalScript_genNL.py)."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--langs",  nargs="+", default=DEFAULT_LANGS,
                        help="Languages. Use ['ruby'] for CSN run.")
    parser.add_argument("--data_root",   default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_root", default="./results")
    parser.add_argument("--max_data",      type=int,   default=MAX_DATA_NUM)
    parser.add_argument("--max_model_len", type=int,   default=4096)
    parser.add_argument("--gpu_util",      type=float, default=0.95)
    args = parser.parse_args()

    MAX_DATA_NUM = args.max_data

    for model_path in args.models:
        model_short_name = os.path.basename(model_path)
        base_tag         = _base_tag(model_path)
        logger.info(f"\n{'=' * 60}\n🚀 Loading Model: {model_short_name}\n{'=' * 60}")

        try:
            llm = LLM(
                model=model_path,
                trust_remote_code=True,
                dtype="bfloat16",
                max_model_len=args.max_model_len,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=args.gpu_util,
                disable_log_stats=True,
            )

            for lang in args.langs:
                logger.info(f"\n🧠 Processing [{lang}] -> {model_short_name}...")
                OUTPUT_FILE = f"{args.output_root}/{model_short_name}/{lang}/{lang}_nl_icse2025_prompt.txt"
                process_language_batch(llm, lang, OUTPUT_FILE,
                                       data_root=args.data_root, base_tag=base_tag)

        except Exception as e:
            logger.error(f"❌ Critical Error with model {model_short_name}: {e}")
            import traceback
            traceback.print_exc()

        # =======================================================
        # 内存清理
        # =======================================================
        try:
            del llm
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()
        try:
            destroy_model_parallel()
        except Exception as e:
            logger.warning(f"Error destroying model parallel: {e}")

        logger.info("♻️ GPU memory released. Ready for next model.")

    logger.info("\n🎉 All tasks completed!")