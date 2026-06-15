"""
Build the per-block structural input for the core-statement-block classifier.

For each sample in the retrieval-augmented JSONL produced by upstream stages, this script:
  1. Reads the translated Python pivot code (`best_python_code` / `python_code` field).
  2. Splits it into AST-aligned semantic blocks via `split_python_by_structure_new`
     (supports nested function definitions — required because LRPL-to-Python translations
     can produce wrapper or inner functions).
  3. Emits per-block `cleaned_seqs` + a structured breakdown
     (function_def / loops / conditionals / assignments / others).
  4. Applies the same split to each `retrieved_candidates[i].code`.

Paper §3.4 ("Core-Statement-Block Extraction Component"): the AST-aligned splitting
described here is what feeds the binary classifier in `classfier_BM25.py`.
"""

import os
import json
import jsonlines
import numpy as np
from tqdm import tqdm

from utils import *   # provides split_python_by_structure_new (and per-language splitters)


def make_retrieved_sentences_structure(input_path, output_path, stop_index=None):
    """
    Process a retrieval-augmented JSONL and emit a structural-block version for the
    core-block classifier.

    Parameters
    ----------
    input_path : str
        JSONL produced by 03_retrieval/BM25.py (each record carries
        `best_python_code` / `python_code` + `retrieved_candidates`).
    output_path : str
        Destination JSONL with the added structural fields.
    stop_index : int or None
        Optional early-stop index (useful for debugging on a small slice).
    """

    def _process_single_code(raw_code):
        """Run `split_python_by_structure_new`, normalise blocks, build the structural dict."""
        if not raw_code:
            return {}, False, True  # result, ast_success, is_empty

        parsed_result, ast_success = split_python_by_structure_new(raw_code)
        code_seqs = []

        if ast_success:
            # 1. function_def is a list (supports nested functions)
            for func_sig in parsed_result.get('function_def', []):
                if func_sig and func_sig.strip():
                    cleaned = ' '.join(func_sig.split()).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

            # 2. other block categories (loops / conditionals / assignments / others)
            for key in ['loops', 'conditionals', 'assignments', 'others']:
                for stmt in parsed_result.get(key, []):
                    cleaned = ' '.join(stmt.split()).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

        structure_info = {
            'cleaned_seqs':  code_seqs,
            'function_def':  parsed_result.get('function_def', []) if ast_success else [],
            'loops':         parsed_result.get('loops', [])        if ast_success else [],
            'conditionals':  parsed_result.get('conditionals', []) if ast_success else [],
            'assignments':   parsed_result.get('assignments', [])  if ast_success else [],
            'others':        parsed_result.get('others', [])       if ast_success else [],
        }
        return structure_info, ast_success, False

    # ----- main loop -----
    total_num = 0
    main_no_ex_seqs_num = 0
    main_ast_failed_num = 0
    main_only_function_def = 0
    retrieved_ast_failed_total = 0

    # pre-count lines for tqdm
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as in_f, \
         jsonlines.open(output_path, mode='w') as out_f:

        for i, line in tqdm(enumerate(in_f), total=total_lines,
                            desc=f"Processing {os.path.basename(input_path)}"):
            if stop_index is not None and i == stop_index:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i}, JSON decode error.")
                continue

            # --- 1. main code (translated Python pivot, NOT raw LRPL) ---
            raw_code = data.get("best_python_code") or data.get("python_code", "")
            structure_info, ast_success, is_empty = _process_single_code(raw_code)
            data.update(structure_info)

            if not ast_success and not is_empty:
                main_ast_failed_num += 1
            if len(data.get("cleaned_seqs", [])) == 0:
                main_no_ex_seqs_num += 1
            if ast_success and len(data.get("cleaned_seqs", [])) == len(data.get("function_def", [])) \
               and len(data.get("cleaned_seqs", [])) != 0:
                main_only_function_def += 1

            # --- 2. retrieved candidates (k-shot examples; code field already Python) ---
            for cand in data.get("retrieved_candidates", []):
                cand_code = cand.get("code", "")
                cand_struct, cand_ast_success, _ = _process_single_code(cand_code)
                cand.update(cand_struct)
                if not cand_ast_success and cand_code:
                    retrieved_ast_failed_total += 1

            out_f.write(data)
            total_num += 1

    # ----- stats -----
    print('=' * 30)
    print(f'Total processed: {total_num}')
    print('--- Main code (translated Python) ---')
    print(f'  AST parse failed: {main_ast_failed_num}')
    print(f'  No cleaned_seqs : {main_no_ex_seqs_num}')
    print(f'  Only function_def: {main_only_function_def}')
    print('--- Retrieved candidates ---')
    print(f'  Total AST failures: {retrieved_ast_failed_total}')
    if total_num > 0:
        print(f'  Main AST fail rate     : {np.round(main_ast_failed_num / total_num, 4)}')
        print(f'  Main no-seqs rate      : {np.round(main_no_ex_seqs_num / total_num, 4)}')
        print(f'  Main only-fn-def rate  : {np.round(main_only_function_def / total_num, 4)}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Add per-block structural fields (cleaned_seqs / function_def / loops / "
                    "conditionals / assignments / others) to every retrieval-augmented record. "
                    "Output is consumed by classfier_BM25.py."
    )
    parser.add_argument("--input",  required=True,
                        help="Input JSONL with `best_python_code` / `python_code` + `retrieved_candidates`")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with added structural fields")
    parser.add_argument("--stop_index", type=int, default=None,
                        help="Optional early-stop index for debugging")
    args = parser.parse_args()

    make_retrieved_sentences_structure(args.input, args.output, args.stop_index)
