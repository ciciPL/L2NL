"""
Unified summary-generation entry point that subsumes all four RQ5 ablation variants
(Full / w/oKey / w/oRet / w/oMod) via three toggles.

Variant matrix
==============
| Variant | USE_SENTENCES | USE_DYNAMIC_BM25_SHOTS | USE_FEW_SHOT |
|---------|:-------------:|:----------------------:|:------------:|
| Full    |     True      |          True          |     True     |
| w/oKey  |     False     |          True          |     True     |
| w/oRet  |     True      |          False         |     True     |
| w/oMod  |     False     |          False         |     False    |

Paper §3.5 + §4.5 RQ5.
"""

import gc
import json
import os
import sys
import torch
import logging
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# Defaults (CLI flags override; see `__main__`)
# ============================================================
MAX_DATA_NUM = None    # default: all; override with --max_data

# ---- 3 toggles that select the ablation variant ----
USE_SENTENCES          = True   # inject `# [KEY LOGIC TRACE]` blocks (§3.4 core-block extractor output)
USE_DYNAMIC_BM25_SHOTS = True   # use BM25-retrieved dynamic shots vs. static FEW_SHOTS_DATA fallback
USE_FEW_SHOT           = True   # include few-shot examples at all
BM25_SHOT_NUM          = 3

DEFAULT_MODELS = [
    "./models/deepseek-coder-1.3b-instruct",
    "./models/deepseek-coder-6.7b-instruct",
    "./models/Llama-3.1-8B-Instruct",
    "./models/Seed-Coder-8B-Instruct",
    "./models/Qwen2.5-Coder-14B-Instruct",
]
DEFAULT_LANGS = ['julia', 'lua', 'ocaml', 'r', 'racket']
DEFAULT_DATA_ROOT = "./data/LowData"

# ============================================================
# Static few-shot fallback (per-language)
# ============================================================
FEW_SHOTS_DATA = {
    "ocaml": [
        {"code": "def make_html_safe(s: str) -> str:\n    return s",
         "trace": ["return s"],
         "summary": "Replace any input in string s to enhance safe."},
        {"code": "def trailing_zeroes(num: int) -> int:\n    if num == 0:\n        return 32\n    p = 0\n    while (num >> p) & 1 == 0:\n        p += 1\n    return p",
         "trace": ["while (num >> p) & 1 == 0", "p += 1"],
         "summary": "Counts trailing zero bits in hashed value using bitwise operations."},
        {"code": "def hamming(first: str, second: str) -> int:\n    if len(first) != len(second):\n        return -1\n    return sum(1 for x, y in zip(first, second) if x != y)",
         "trace": ["if len(first) != len(second)", "sum(1 for x, y in zip(first, second) if x != y)"],
         "summary": "Calculates Hamming distance between two strings of equal length."},
    ],
    "r": [
        {"code": "def PowerStack_Calc(Power, N):\n    result = N * Power\n    return result",
         "trace": ["result = N * Power"],
         "summary": "Calculates power stack by multiplying single cell power by number of cells."},
        {"code": "def trackOpposite(f_tr):\n    l_d = f_tr + 180.0\n    while l_d >= 360.0:\n        l_d = l_d - 360.0\n    return l_d",
         "trace": ["l_d = f_tr + 180.0", "while l_d >= 360.0: l_d = l_d - 360.0"],
         "summary": "Function to calculate the opposite direction by adding 180 degrees and normalizing to 0-360 range."},
        {"code": "import re\ndef normalize_meaning(source_string):\n    source_string = source_string.replace('\"', '&quot;')\n    source_string = source_string.replace(\"'\", '&apos;')\n    source_string = source_string.replace('\\n', ' ')\n    source_string = source_string.replace('\\t', ' ')\n    source_string = re.sub(r'<.*?>', '', source_string)\n    source_string = re.sub(r' {2,}', ' ', source_string)\n    return source_string.strip()",
         "trace": ["source_string.replace('\"', '&quot;')", "re.sub('<.*?>', '', source_string)", "source_string.strip()"],
         "summary": "Normalizes a string by escaping HTML tags, replacing quotes, removing newlines, tabs, and extra spaces, and trimming whitespace."},
    ],
    "julia": [
        {"code": "import os\ndef get_id_from_repo(repo_dir):\n    return os.path.basename(repo_dir)",
         "trace": ["return os.path.basename(repo_dir)"],
         "summary": "Extracts Google Drive File/Folder ID from repository directory path."},
        {"code": "def merge_segdb(segdbs):\n    return [item for sublist in segdbs for item in sublist]",
         "trace": ["return [item for sublist in segdbs for item in sublist]"],
         "summary": "Concatenates a list of segdbs into a single segdb."},
        {"code": "def linear_cell_to_tuple(c1, repeat_units):\n    c1x = c1 % repeat_units[0]\n    c1y = ((c1 - c1x) // repeat_units[0]) % repeat_units[1]\n    c1z = (c1 - c1x - c1y * repeat_units[0]) // (repeat_units[0] * repeat_units[1])\n    return c1x, c1y, c1z",
         "trace": ["c1x = c1 % repeat_units[0]", "c1y = ((c1 - c1x) // repeat_units[0]) % repeat_units[1]"],
         "summary": "Converts a linear index to a 3D tuple using lexicographic indexing."},
    ],
    "lua": [
        {"code": "def _translate_message(message):\n    if message:\n        return {'id': message.id, 'project_id': message.project_id}\n    else:\n        return None",
         "trace": ["def _translate_message(message)", "return { \"id\": message.id, ... }"],
         "summary": "Translates Message model to a dictionary."},
        {"code": "def compare_rule_hits_count(r1, r2, diff):\n    if r1 is None or r2 is None:\n        return False\n    r1_count = r1['report']['meta']['count']\n    return r1_count == r2['report']['meta']['count']",
         "trace": ["r1_count = r1.get('report').get('meta').get('count')", "return r1_count == r2_count"],
         "summary": "Compares rule hit counts and updates diff structure."},
        {"code": "def getBoundsOverlap(bb1, bb2):\n    x1, y1, x2, y2 = bb1[0], bb1[1], bb1[2], bb1[3]\n    x3, y3, x4, y4 = bb2[0], bb2[1], bb2[2], bb2[3]\n    minX = max(x1, x3); minY = max(y1, y3); maxX = min(x2, x4); maxY = min(y2, y4)\n    if maxX < minX or maxY < minY: return False\n    return [[minX, minY, maxX, maxY]]",
         "trace": ["minX = max(x1, x3); maxX = min(x2, x4)", "if maxX < minX or maxY < minY: return False"],
         "summary": "Calculates intersection of two bounding boxes."},
    ],
    "racket": [
        {"code": "import math\ndef get_color_towards(color1, color2, fraction):\n    return [math.floor(x + fraction * (y - x)) for x, y in zip(color1, color2)]",
         "trace": ["math.floor(x + fraction * (y - x))"],
         "summary": "Calculates a color between two given colors based on a fraction."},
        {"code": "def _map_boolean_to_human_readable(boolean, resource, token):\n    if boolean:\n        return \"Yes\"\n    else:\n        return \"No\"",
         "trace": ["if boolean: return \"Yes\"", "return \"No\""],
         "summary": "Maps boolean to Yes or No for human readability."},
        {"code": "def on_segment(p, q, r):\n    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and\n            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))",
         "trace": ["q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0])"],
         "summary": "Checks if point q lies on line segment pr for collinear points p, q, r."},
    ],
}


# ============================================================
# Data loading (load_merged_data) — minimal IO based on toggles
# ============================================================
def load_merged_data(lang, max_num=None, data_root=None, base_tag="qwen3th"):
    """
    Loads the per-language inputs. The set of files loaded depends on toggles:
      - code_file is always required (translated pivot Python).
      - struct_file is required only when USE_SENTENCES is True.
      - bm25_file  is required only when USE_DYNAMIC_BM25_SHOTS is True.
    """
    root = data_root or DEFAULT_DATA_ROOT
    base_dir = f"{root}/{lang}/trans_{base_tag}"
    code_file   = os.path.join(base_dir, f"{lang}_python_best_candidate_vllm_B0.5_S0.5.jsonl")
    struct_file = os.path.join(base_dir, f"python_{lang}_structure_B0.5_S0.5_preds.jsonl")
    bm25_file   = os.path.join(base_dir, f"python_{lang}_BM25_results_sentences_preds.jsonl")

    if not os.path.exists(code_file):
        logger.error(f"Missing code file for {lang}: {code_file}")
        return []

    need_struct = USE_SENTENCES
    if need_struct and not os.path.exists(struct_file):
        logger.error(f"USE_SENTENCES=True but struct file missing: {struct_file}")
        return []

    need_bm25 = USE_FEW_SHOT and USE_DYNAMIC_BM25_SHOTS
    bm25_exists = os.path.exists(bm25_file) if need_bm25 else False
    if need_bm25 and not bm25_exists:
        logger.warning(f"BM25 shots enabled but file not found: {bm25_file}. Falling back to static shots.")

    with open(code_file, 'r', encoding='utf-8') as f_c:
        code_lines = f_c.readlines()
    struct_lines = []
    bm25_lines   = []
    if need_struct:
        with open(struct_file, 'r', encoding='utf-8') as f_s:
            struct_lines = f_s.readlines()
    if need_bm25 and bm25_exists:
        with open(bm25_file, 'r', encoding='utf-8') as f_b:
            bm25_lines = f_b.readlines()

    limit = len(code_lines)
    if need_struct: limit = min(limit, len(struct_lines))
    if need_bm25 and bm25_exists: limit = min(limit, len(bm25_lines))
    if max_num: limit = min(limit, max_num)

    merged = []
    for i in range(limit):
        try:
            c_item = json.loads(code_lines[i])
        except json.JSONDecodeError:
            continue

        # snippets (only when USE_SENTENCES)
        snippets = []
        if need_struct:
            try:
                s_item = json.loads(struct_lines[i])
            except json.JSONDecodeError:
                continue
            snippets = s_item.get("cleaned_seqs_pred", [])
            if not snippets or (isinstance(snippets, list)
                                and not any(s.strip() for s in snippets if isinstance(s, str))):
                func_def = s_item.get("function_def", "")
                if func_def and isinstance(func_def, str) and func_def.strip():
                    snippets = [func_def]
                else:
                    snippets = []

        # bm25 candidates (only when need_bm25)
        bm25_candidates = []
        if need_bm25 and bm25_exists:
            try:
                b_item = json.loads(bm25_lines[i])
                bm25_candidates = b_item.get("retrieved_candidates", [])
            except json.JSONDecodeError:
                bm25_candidates = []

        merged.append({
            "idx": i + 1,
            "original_id": c_item.get("index") or c_item.get("id"),
            "code": c_item.get("best_python_code", ""),
            "snippets": snippets,
            "bm25_candidates": bm25_candidates,
        })

    logger.info(f"Loaded {len(merged)} items for {lang} "
                f"(SENT={USE_SENTENCES}, DYN={need_bm25 and bm25_exists}, FEW={USE_FEW_SHOT})")
    return merged


# ============================================================
# Prompt construction
# ============================================================
def clean_extracted_summary(raw_text):
    content = raw_text.split("<SUMMARY>")[-1] if "<SUMMARY>" in raw_text else raw_text
    content = content.replace("</SUMMARY>", "").strip()
    if content.startswith("#"):
        content = content.lstrip("#").strip()
    return content


def format_block_annotation_style(code, snippets, summary=None, return_parts=False):
    """
    Build the User/Assistant text pair for one example.
    If USE_SENTENCES is False, the [KEY LOGIC TRACE] section is omitted.
    """
    text  = "# ==================================================\n"
    text += "# [USER INPUT CODE]\n"
    text += f"{code.strip()}\n\n"

    if USE_SENTENCES:
        formatted = []
        if isinstance(snippets, list):
            for snip in snippets:
                if snip and isinstance(snip, str) and snip.strip():
                    formatted.append(f"# > {snip.strip()}")
        elif isinstance(snippets, str) and snippets.strip():
            formatted.append(f"# > {snippets.strip()}")
        trace_content = "\n".join(formatted)
        text += "# [KEY LOGIC TRACE]\n"
        text += (f"{trace_content}\n\n" if trace_content else "\n")

    text += "Concisely summarize the code provided in 1-3 sentences.\n"

    assistant_text = f"<SUMMARY>\n{summary.strip()}\n</SUMMARY>" if summary else ""

    if return_parts:
        return text, assistant_text
    return text + (assistant_text + "\n" if assistant_text else "")


def build_completion_prompt(lang, code, snippets, dynamic_candidates=None):
    """
    Completion-style prompt: System guidelines + (optional) few-shot examples + target user block.
    Single text string (no chat template) — works with chat and base models alike.
    """
    text = """You're a specialized AI assisting with Python code summaries, deeply knowledgeable in computer science.

GUIDELINES:
1. Code Analysis: Read the source code under '# [USER INPUT CODE]'.
"""
    if USE_SENTENCES:
        text += "2. Trace Usage: The '# [KEY LOGIC TRACE]' section provides execution hints. Use it as an auxiliary aid to understand the core functionality, but avoid describing low-level implementation details (e.g., specific variable manipulations) in your summary. Focus on the high-level intent.\n"
        text += "3. Output Format: Output your natural language summary inside <SUMMARY> and </SUMMARY> tags.\n"
    else:
        text += "2. Output Format: Output your natural language summary inside <SUMMARY> and </SUMMARY> tags.\n"

    text += "\nExamples:\n\n"

    # ---- (A) Few-shot ----
    if USE_FEW_SHOT:
        examples_to_use = []
        if USE_DYNAMIC_BM25_SHOTS and dynamic_candidates:
            for cand in dynamic_candidates[:BM25_SHOT_NUM]:
                examples_to_use.append({
                    "code":    cand.get("code", ""),
                    "trace":   cand.get("cleaned_seqs_pred", []) if USE_SENTENCES else [],
                    "summary": cand.get("summary", ""),
                })
        elif lang in FEW_SHOTS_DATA:
            for ex in FEW_SHOTS_DATA[lang]:
                examples_to_use.append({
                    "code":    ex["code"],
                    "trace":   ex["trace"] if USE_SENTENCES else [],
                    "summary": ex["summary"],
                })

        for ex in examples_to_use:
            user_part, assistant_part = format_block_annotation_style(
                ex['code'], ex['trace'], ex['summary'], return_parts=True
            )
            text += user_part.strip() + "\n"
            text += assistant_part.strip() + "\n\n"

    # ---- (B) Target ----
    target_user_part, _ = format_block_annotation_style(
        code, snippets, summary=None, return_parts=True
    )
    text += target_user_part.strip() + "\n"
    text += "<SUMMARY>\n"
    return text


def variant_tag():
    """Short tag used in output file names."""
    if USE_SENTENCES and USE_DYNAMIC_BM25_SHOTS and USE_FEW_SHOT:
        return "full"
    if (not USE_SENTENCES) and USE_DYNAMIC_BM25_SHOTS and USE_FEW_SHOT:
        return "woKey"
    if USE_SENTENCES and (not USE_DYNAMIC_BM25_SHOTS) and USE_FEW_SHOT:
        return "woRet"
    if (not USE_SENTENCES) and (not USE_DYNAMIC_BM25_SHOTS) and (not USE_FEW_SHOT):
        return "woMod"
    return f"sent{int(USE_SENTENCES)}_dyn{int(USE_DYNAMIC_BM25_SHOTS)}_few{int(USE_FEW_SHOT)}"


# ============================================================
# Inference per language
# ============================================================
def process_language_batch(llm, lang, output_txt_file, data_root=None, base_tag="qwen3th"):
    debug_file = output_txt_file.replace(".txt", "_debug.jsonl")

    data = load_merged_data(lang, max_num=MAX_DATA_NUM, data_root=data_root, base_tag=base_tag)
    if not data:
        return

    prompts = []
    for item in data:
        prompt = build_completion_prompt(
            lang, item['code'], item['snippets'],
            dynamic_candidates=item.get('bm25_candidates'),
        )
        prompts.append(prompt)
        item['final_prompt'] = prompt

    sampling_params = SamplingParams(
        temperature=0.0, top_p=0.95, top_k=50, max_tokens=128, n=1,
        stop=['</SUMMARY>', '<|eot_id|>', '<|end_of_text|>'],
    )

    logger.info(f"Running inference for {len(prompts)} items (variant={variant_tag()})...")
    outputs = llm.generate(prompts, sampling_params)

    os.makedirs(os.path.dirname(output_txt_file), exist_ok=True)
    with open(output_txt_file, 'w', encoding='utf-8') as f_out, \
         open(debug_file, 'w', encoding='utf-8') as f_debug:
        for i, o in enumerate(outputs):
            item = data[i]
            generated = o.outputs[0].text
            clean = clean_extracted_summary(generated).replace('\n', ' ')
            f_out.write(f"{item['idx']}\t{clean}\n")
            f_debug.write(json.dumps({
                "index": item['idx'],
                "original_id": item['original_id'],
                "lang": lang,
                "variant": variant_tag(),
                "prompt": item['final_prompt'],
                "raw_output": generated,
                "clean_summary": clean,
            }, ensure_ascii=False) + '\n')
    logger.info(f"✅ Results saved to {output_txt_file}")


# ============================================================
# Main
# ============================================================
def _base_tag(model_path: str) -> str:
    """Server-side experimental folder convention for paths."""
    if 'Llama' in model_path: return 'Llama3th'
    if 'Seed'  in model_path: return 'Seed3th'
    if 'deepseek-coder-6.7b-instruct' in model_path: return 'ds_mid3th'
    if '1.3b'  in model_path: return 'ds_small3th'
    if 'Qwen'  in model_path: return 'qwen3th'
    return os.path.basename(model_path.rstrip('/'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified summary-generation script (Full / w/oKey / w/oRet / w/oMod). "
                    "Variant selected via the three --use_* toggles; see module docstring."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="HuggingFace model paths.")
    parser.add_argument("--langs",  nargs="+", default=DEFAULT_LANGS,
                        help="Languages to process. Use ['ruby'] for the CSN run.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                        help="Root containing per-lang trans_<base>/... inputs (default: ./data/LowData).")
    parser.add_argument("--output_root", default="./results",
                        help="Where to write <model>/<lang>/<lang>_nl_<variant>.txt (default: ./results).")
    parser.add_argument("--use_sentences",    dest="use_sentences",    action="store_true", default=USE_SENTENCES,
                        help="Inject [KEY LOGIC TRACE] core-block hints (default: on).")
    parser.add_argument("--no_sentences",     dest="use_sentences",    action="store_false",
                        help="Disable [KEY LOGIC TRACE] (→ w/oKey ablation).")
    parser.add_argument("--use_dynamic_bm25", dest="use_dyn",          action="store_true", default=USE_DYNAMIC_BM25_SHOTS,
                        help="Use BM25-retrieved dynamic few-shot examples (default: on).")
    parser.add_argument("--no_dynamic_bm25",  dest="use_dyn",          action="store_false",
                        help="Use static FEW_SHOTS_DATA examples instead (→ w/oRet ablation).")
    parser.add_argument("--use_few_shot",     dest="use_few_shot",     action="store_true", default=USE_FEW_SHOT,
                        help="Include few-shot examples (default: on).")
    parser.add_argument("--no_few_shot",      dest="use_few_shot",     action="store_false",
                        help="Zero-shot prompt (→ w/oMod ablation).")
    parser.add_argument("--bm25_shot_num", type=int, default=BM25_SHOT_NUM,
                        help="How many BM25 shots to use (default: 3).")
    parser.add_argument("--max_data",      type=int, default=MAX_DATA_NUM,
                        help="Cap samples per language (default: all).")
    parser.add_argument("--max_model_len", type=int, default=9192)
    parser.add_argument("--gpu_util",      type=float, default=0.95)
    args = parser.parse_args()

    # Apply CLI overrides to module-level toggles (read by load_merged_data /
    # build_completion_prompt / format_block_annotation_style / variant_tag).
    USE_SENTENCES          = args.use_sentences
    USE_DYNAMIC_BM25_SHOTS = args.use_dyn
    USE_FEW_SHOT           = args.use_few_shot
    BM25_SHOT_NUM          = args.bm25_shot_num
    MAX_DATA_NUM           = args.max_data

    tag = variant_tag()
    logger.info(f"Variant: {tag}  "
                f"(USE_SENTENCES={USE_SENTENCES}, USE_DYNAMIC_BM25_SHOTS={USE_DYNAMIC_BM25_SHOTS}, "
                f"USE_FEW_SHOT={USE_FEW_SHOT})")

    for model_path in args.models:
        model_short = os.path.basename(model_path)
        base_tag    = _base_tag(model_path)
        logger.info(f"\n{'=' * 60}\n🚀 Loading model: {model_short}\n{'=' * 60}")

        try:
            llm = LLM(
                model=model_path, trust_remote_code=True, dtype="bfloat16",
                max_model_len=args.max_model_len,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=args.gpu_util,
                disable_log_stats=True,
            )
            for lang in args.langs:
                OUTPUT_FILE = f"{args.output_root}/{model_short}/{lang}/{lang}_nl_{tag}.txt"
                process_language_batch(llm, lang, OUTPUT_FILE,
                                       data_root=args.data_root, base_tag=base_tag)
        except Exception as e:
            logger.error(f"❌ Critical error with model {model_short}: {e}")
            import traceback; traceback.print_exc()

        try: del llm
        except: pass
        gc.collect()
        torch.cuda.empty_cache()
        try: destroy_model_parallel()
        except Exception as e: logger.warning(f"Error destroying model parallel: {e}")
        logger.info("♻️ GPU memory released. Ready for next model.")
