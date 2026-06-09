import json
import os  # <--- 添加导入 os 库
import jsonlines
import numpy as np
from tqdm import tqdm

import rouge_not_a_wrapper as my_rouge
from identifier_splitting import split_identifier_into_parts
from utils import *

ex_codes_num = []
ex_words_num = []


def split_code_to_blocks(code_token_list):
    first_lf_brace = True
    lf_brace_up = 0
    lf_bracket_up = 0
    idxs = []
    end_idx = 0
    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_brace_up and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{' and not first_lf_brace:
            lf_brace_up += 1
        elif i == '}' and (lf_brace_up == 1):
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            lf_brace_up -= 1
        elif i == '}' and (lf_brace_up != 1):
            lf_brace_up -= 1
        elif i == '{' and first_lf_brace:
            idxs.append((0, idx))
            end_idx = idx + 1
            first_lf_brace = False
    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if idx == 0:
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs


def split_java_to_seqs(code_token_list):
    lf_bracket_up = 0
    idxs = []
    end_idx = 0
    for idx, i in enumerate(code_token_list):
        if i == ';' and not lf_bracket_up:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '(':
            lf_bracket_up += 1
        elif i == ')':
            lf_bracket_up -= 1
        elif i == '{':
            idxs.append((end_idx, idx))
            end_idx = idx + 1
        elif i == '}':
            end_idx = idx + 1
    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if code_snap and code_snap[-1] == '{':  # <--- 增加检查 code_snap 是否为空
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs


def get_code_ex(code_seqs, nl_seq):
    if len(code_seqs) == 0 or len(nl_seq) == 0:
        return [], [], [], [], [], None

    global ex_codes_num
    global ex_words_num
    fs = []
    ps = []
    rs = []
    for i, code_seq in enumerate(code_seqs):
        # 增加检查，确保 code_seq 不是空字符串
        if not code_seq.strip():
            fs.append(0.0)
            ps.append(0.0)
            rs.append(0.0)
            continue
        rouge_l_f, rouge_l_p, rouge_l_r = my_rouge.rouge_l_summary_level([code_seq], nl_seq)
        fs.append(rouge_l_f)
        ps.append(rouge_l_p)
        rs.append(rouge_l_r)

    scores = np.array(rs)
    id_sort_by_scores = np.argsort(scores, kind='stable')[::-1]
    max_Rouge_l_r = 0.0
    ex_ids = []
    ex_codes = []

    for i in range(len(code_seqs)):
        current_best_id = id_sort_by_scores[i]
        # 增加检查，确保当前考虑的 code_seq 不是空字符串
        if not code_seqs[current_best_id].strip():
            continue

        new_ex_ids = sorted(ex_ids + [current_best_id])
        new_ex_codes = [code_seqs[idx] for idx in new_ex_ids]
        _, _, Rouge_l_r = my_rouge.rouge_l_summary_level(new_ex_codes, nl_seq)

        if Rouge_l_r > max_Rouge_l_r:
            ex_ids = new_ex_ids
            ex_codes = new_ex_codes
            max_Rouge_l_r = Rouge_l_r

    ex_codes_num.append(len(ex_codes))
    ex_words = ' '.join(ex_codes).split(' ')
    ex_words_num.append(len(ex_words))
    return ex_codes, ex_ids, fs, ps, rs, max_Rouge_l_r


def make_pcsd_dataset(input_path, output_path, language):
    total_num = 0
    no_ex_seqs_num = 0
    ast_failed_num = 0  # <--- 添加 AST 失败计数器

    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for line in tqdm(in_f, total=total_lines, desc=f"Processing PCSD {os.path.basename(input_path)}"):
            js = json.loads(line.strip())

            idx = js['id']
            raw_code = js['raw_code']
            nl_tokens = js['comment'].split()
            code_tokens_full = js['code'].split()

            # 1. 使用 AST 分割代码，并获取成功标志
            code_statement_strings, ast_success = split_python_with_ast(raw_code)  # <--- 接收标志
            if not ast_success:
                ast_failed_num += 1  # <--- 如果失败，增加计数

            # 2. 对每个分割出的语句进行处理
            code_seqs = []
            for stmt_str in code_statement_strings:
                stmt_tokens = stmt_str.split()
                stmt_split_ids = split_identifier_into_parts(' '.join(stmt_tokens))
                stmt_lower = ' '.join(stmt_split_ids).lower()
                if stmt_lower.strip():
                    code_seqs.append(stmt_lower)

            if not code_seqs:
                code_seqs = [' '.join([s.lower() for s in split_identifier_into_parts(' '.join(code_tokens_full))])]

            # ... (后续处理 NL, code_lower, 获取标签等) ...
            nl = ' '.join(nl_tokens).replace('\n', '')
            nl = split_identifier_into_parts(nl)
            nl_seq = [' '.join(nl).lower()]

            code_full_str = ' '.join(code_tokens_full).replace('\n', '')
            code_full_split = split_identifier_into_parts(code_full_str)
            code_lower = [s.lower() for s in code_full_split]

            out_js = {'idx': idx}
            out_js['raw_code'] = raw_code
            out_js['cleaned_nl'] = nl_seq
            out_js['cleaned_codes'] = code_lower
            out_js['cleaned_blocks'] = []
            out_js['cleaned_blocks_ex'] = []

            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_seqs, nl_seq)

            if len(ex_seqs) == 0:
                ex_seqs = [' '.join(code_lower)]
                no_ex_seqs_num += 1
                out_js['ex_labels'] = [0] * len(code_seqs) if code_seqs else [0]
            else:
                out_js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            out_js['fs'] = fs
            out_js['ps'] = ps
            out_js['rs'] = rs
            out_js['max_Rouge_l_r'] = max_Rouge_l_r
            out_js['cleaned_seqs'] = code_seqs
            out_js['cleaned_seqs_ex'] = ex_seqs
            if 'ex_labels' not in out_js:
                out_js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    print('AST parse failed num:', ast_failed_num)  # <--- 打印统计结果
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
        print('AST parse failed %:', np.round(ast_failed_num / total_num, 4))  # <--- 打印百分比
    else:
        print('no ex seqs %: 0.0')
        print('AST parse failed %: 0.0')


def make_pcsd_dataset_structure(input_path, output_path, language):
    total_num = 0
    no_ex_seqs_num = 0
    ast_failed_num = 0  # AST 失败计数器

    total_lines = sum(1 for _ in open(input_path, encoding="utf-8"))

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for line in tqdm(in_f, total=total_lines, desc=f"Processing PCSD {os.path.basename(input_path)}"):
            js = json.loads(line.strip())

            idx = js['id']
            raw_code = js['raw_code']
            nl_tokens = js['comment'].split()
            code_tokens_full = js['code'].split()

            # ✅ 使用结构化 AST 分割代码
            struct_blocks, ast_success = split_python_by_structure(raw_code)
            if not ast_success:
                ast_failed_num += 1

            # ✅ 合并所有结构块
            code_blocks = []
            if struct_blocks['function_def']:
                code_blocks.append(struct_blocks['function_def'])
            code_blocks.extend(struct_blocks['loops'])
            code_blocks.extend(struct_blocks['conditionals'])
            code_blocks.extend(struct_blocks['assignments'])
            code_blocks.extend(struct_blocks['others'])

            # ✅ 每个结构块分词处理
            code_seqs = []
            for stmt_str in code_blocks:
                stmt_tokens = stmt_str.split()
                stmt_split_ids = split_identifier_into_parts(' '.join(stmt_tokens))
                stmt_lower = ' '.join(stmt_split_ids).lower()
                if stmt_lower.strip():
                    code_seqs.append(stmt_lower)

            # ✅ 若为空，fallback 为整个代码
            if not code_seqs:
                code_seqs = [' '.join([s.lower() for s in split_identifier_into_parts(' '.join(code_tokens_full))])]

            # ✅ 自然语言摘要处理
            nl = ' '.join(nl_tokens).replace('\n', '')
            nl = split_identifier_into_parts(nl)
            nl_seq = [' '.join(nl).lower()]

            # ✅ 全代码处理（未分段）
            code_full_str = ' '.join(code_tokens_full).replace('\n', '')
            code_full_split = split_identifier_into_parts(code_full_str)
            code_lower = [s.lower() for s in code_full_split]

            # ✅ 输出字段构造
            out_js = {'idx': idx}
            out_js['cleaned_nl'] = nl_seq
            out_js['cleaned_codes'] = code_lower
            out_js['cleaned_blocks'] = []  # 预留字段
            out_js['cleaned_blocks_ex'] = []

            # ✅ 获取与摘要最匹配的片段（ROUGE）
            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_seqs, nl_seq)

            if len(ex_seqs) == 0:
                ex_seqs = [' '.join(code_lower)]
                no_ex_seqs_num += 1
                out_js['ex_labels'] = [0] * len(code_seqs) if code_seqs else [0]
            else:
                out_js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            out_js['fs'] = fs
            out_js['ps'] = ps
            out_js['rs'] = rs
            out_js['max_Rouge_l_r'] = max_Rouge_l_r
            out_js['cleaned_seqs'] = code_seqs
            out_js['cleaned_seqs_ex'] = ex_seqs
            out_js['raw_code'] = raw_code

            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    print('AST parse failed num:', ast_failed_num)
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
        print('AST parse failed %:', np.round(ast_failed_num / total_num, 4))
    else:
        print('no ex seqs %: 0.0')
        print('AST parse failed %: 0.0')


def make_py_dataset(input_path, output_path, language):
    total_num = 0
    no_ex_blocks_num = 0
    no_ex_seqs_num = 0

    # --- 开始修改 ---
    # 1. 计算总行数
    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    # --- 结束修改 ---

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        # --- 开始修改 ---
        # 2. 将 total_lines 传递给 tqdm，并添加 desc 参数
        for idx, line in tqdm(enumerate(in_f), total=total_lines, desc=f"Processing {os.path.basename(input_path)}"):
            # --- 结束修改 ---
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx

            code_tokens = js['code_tokens']
            codes = ' '.join(code_tokens).replace('\n', '')
            code_tokens = codes.split()
            # todo 这里需要根据语言修改函数
            code_seqs = split_python_to_seqs(code_tokens)
            code_seqs = ' <spt> '.join(code_seqs)

            code_seqs = split_identifier_into_parts(code_seqs)
            code_seqs = ' '.join(code_seqs)
            code_seqs_lower = code_seqs.lower()
            code_seqs = code_seqs_lower.split(' <spt> ')
            # 增加过滤空字符串的步骤
            code_seqs = [seq for seq in code_seqs if seq.strip()]

            code = ' '.join(code_tokens).replace('\n', '')
            code = split_identifier_into_parts(code)
            code_lower = [s.lower() for s in code]

            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = split_identifier_into_parts(nl)
            nl_seq = [' '.join(nl).lower()]

            js['cleaned_nl'] = nl_seq
            js['cleaned_codes'] = code_lower
            # ex_blocks, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_blocks, nl_seq)
            # if len(ex_blocks) == 0:
            #     ex_blocks = code_lower
            #     no_ex_blocks_num += 1
            js['cleaned_blocks'] = []
            js['cleaned_blocks_ex'] = []

            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_seqs, nl_seq)

            # 检查 ex_seqs 是否为空，如果为空，则使用 code_lower 并增加计数
            if len(ex_seqs) == 0:
                ex_seqs = [' '.join(code_lower)]  # 使用列表包装，保持数据类型一致
                no_ex_seqs_num += 1
                # 如果没有找到 ex_seqs，ex_ids 应该是空的，所以标签都是 0
                js['ex_labels'] = [0] * len(code_seqs)
            else:
                js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            js['cleaned_seqs'] = code_seqs
            js['cleaned_seqs_ex'] = ex_seqs
            # 确保即使没有找到ex_seqs，也有标签（尽管可能都是0）
            if 'ex_labels' not in js:
                js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            out_f.write(js)

            total_num += 1

    print('total num:', total_num)
    print('no ex blocks num:', no_ex_blocks_num)
    print('no ex seqs num:', no_ex_seqs_num)

    # 确保 total_num 不是 0 以避免除零错误
    if total_num > 0:
        print('no ex blocks %:', np.round(no_ex_blocks_num / total_num, 4))
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
    else:
        print('no ex blocks %: 0.0')
        print('no ex seqs %: 0.0')


def make_pcsd_dataset_modified(input_path, output_path):
    total_processed_num = 0
    ast_success_num = 0
    ast_failed_num = 0

    # 计算总行数以用于 tqdm 进度条
    total_lines = 0
    try:
        with open(input_path, encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
    except FileNotFoundError:
        print(f"错误: 输入文件 {input_path} 未找到。")
        return

    with open(input_path, encoding="utf-8") as in_f, \
            jsonlines.open(output_path, mode='w') as out_f:  # 使用 jsonlines 写入

        for line_number, line_content in enumerate(
                tqdm(in_f, total=total_lines, desc=f"处理 {os.path.basename(input_path)}"), 1):
            line = line_content.strip()
            if not line:  # 跳过空行
                continue

            raw_code = ""  # 初始化
            idx_str = f"line_{line_number}"  # 默认 idx 以防行格式错误

            try:
                # 期望格式 "索引:代码"
                parts = line.split(':', 1)
                if len(parts) == 2:
                    idx_str, raw_code = parts
                    idx = idx_str.strip()
                else:
                    # 如果行不包含':'，则将整行视为代码，并使用行号作为索引
                    # 或者您可以选择标记为错误
                    print(f"警告: 第 {line_number} 行格式不正确 (缺少':')，将整行视为代码: {line}")
                    raw_code = line
                    idx = idx_str  # 使用默认行号索引
                    # 如果希望将格式错误的行严格计为AST失败：
                    # ast_failed_num += 1
                    # total_processed_num += 1
                    # error_entry = {"idx": idx, "raw_code": line, "error": "Malformed input line (no colon)", "ast_success": False, "cleaned_seqs": []}
                    # out_f.write(error_entry)
                    # continue
            except Exception as e:  # 捕获分割过程中的其他潜在错误
                print(f"警告: 解析第 {line_number} 行时出错: {line}. 错误: {e}")
                ast_failed_num += 1
                total_processed_num += 1
                error_entry = {"idx": idx_str, "raw_code": line, "error": f"Line parsing error: {e}",
                               "ast_success": False, "cleaned_seqs": []}
                out_f.write(error_entry)
                continue

            total_processed_num += 1
            out_js = {'idx': idx, 'raw_code': raw_code}

            # 1. 使用 AST 分割代码，并获取成功标志
            code_statement_strings, ast_success = split_python_with_ast(raw_code)

            if ast_success:
                ast_success_num += 1
                out_js['ast_success'] = True

                # 2. 对每个分割出的语句进行处理
                cleaned_seqs = []
                if code_statement_strings:  # 确保列表不是 None 或空的
                    for stmt_str in code_statement_strings:
                        stmt_str_cleaned = stmt_str.strip()
                        if not stmt_str_cleaned:  # 跳过处理后为空的语句
                            continue

                        # 使用 split_identifier_into_parts 处理语句字符串
                        stmt_split_ids = split_identifier_into_parts(stmt_str_cleaned)
                        # 将分割后的部分连接起来并转为小写
                        stmt_lower = ' '.join(stmt_split_ids).lower()

                        if stmt_lower.strip():  # 确保结果不为空白
                            cleaned_seqs.append(stmt_lower)
                out_js['cleaned_seqs'] = cleaned_seqs
            else:
                ast_failed_num += 1
                out_js['ast_success'] = False
                out_js['cleaned_seqs'] = []  # AST 失败则没有分割的语句

            out_f.write(out_js)

    print(f"\n--- {os.path.basename(input_path)} 的统计信息 ---")
    print(f"总共处理的条目数: {total_processed_num}")
    print(f"AST 解析成功数: {ast_success_num}")
    print(f"AST 解析失败数: {ast_failed_num}")

    if total_processed_num > 0:
        success_percentage = np.round((ast_success_num / total_processed_num) * 100, 2)
        failure_percentage = np.round((ast_failed_num / total_processed_num) * 100, 2)
        print(f"AST 解析成功率 %: {success_percentage}%")
        print(f"AST 解析失败率 %: {failure_percentage}%")
    else:
        print("AST 解析成功率 %: 0.0%")
        print("AST 解析失败率 %: 0.0%")


def make_LRPL_sentences_ast(input_path, output_path, stop_index):
    total_num = 0
    no_ex_seqs_num = 0
    ast_failed_num = 0  # <--- 添加 AST 失败计数器

    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for line in tqdm(in_f, total=total_lines, desc=f"Processing LRPL {os.path.basename(input_path)}"):

            idx = line.split(':')[0].strip()
            raw_code = line.split(':')[1].strip()
            if int(idx) == stop_index: break
            # 1. 使用 AST 分割代码，并获取成功标志
            code_statement_strings, ast_success = parse_R.split_r_into_statements(raw_code,
                                                                                  '../../../../vendor/parse_fixed.R')  # <--- 接收标志

            # 2. 对每个分割出的语句进行处理
            code_seqs = []
            if ast_success:
                for stmt_str in code_statement_strings:
                    stmt_tokens = stmt_str.split()
                    stmt_split_ids = split_identifier_into_parts(' '.join(stmt_tokens))
                    stmt_lower = ' '.join(stmt_split_ids).lower()
                    if stmt_lower.strip():
                        code_seqs.append(stmt_lower)
            else:
                ast_failed_num += 1  # <--- 如果失败，增加计数
                # code_seqs=['Parse_faith']

            if len(code_statement_strings) == 1:
                no_ex_seqs_num += 1
                # code_seqs=['len = 1']

            out_js = {'idx': idx}
            out_js['raw_codes'] = raw_code
            out_js['cleaned_seqs'] = code_seqs
            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    print('AST parse failed num:', ast_failed_num)  # <--- 打印统计结果
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
        print('AST parse failed %:', np.round(ast_failed_num / total_num, 4))  # <--- 打印百分比
    else:
        print('no ex seqs %: 0.0')
        print('AST parse failed %: 0.0')


def make_LRPL_sentences_structure(input_path, output_path, stop_index, r_script_path):
    total_num = 0
    no_ex_seqs_num = 0
    ast_failed_num = 0
    only_fuction_def = 0
    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for line in tqdm(in_f, total=total_lines, desc=f"Processing LRPL {os.path.basename(input_path)}"):

            idx = line.split(':', 1)[0].strip()
            # --- START: 修复代码 ---
            # 1. 读取包含 "\\n" 的单行代码
            raw_code_line = line.split(':', 1)[1].strip()

            # 2. 将 "\\n" (两个字符) 替换回 "\n" (一个换行符)
            raw_code = raw_code_line.encode('latin-1').decode('unicode_escape')
            # --- END: 修复代码 ---
            if int(idx) == stop_index:
                break

            parsed_result, ast_success = split_python_by_structure(raw_code)

            code_seqs = []
            if ast_success:
                # 1. 单独处理 function_def (它是一个字符串)
                func_def_str = parsed_result.get('function_def', '')
                if func_def_str and func_def_str.strip():
                    tokens = func_def_str.split()
                    cleaned = ' '.join(tokens).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

                # 2. 循环处理其他键 (它们是列表)
                for key in ['loops', 'conditionals', 'assignments', 'others']:
                    for stmt in parsed_result.get(key, []):
                        tokens = stmt.split()
                        cleaned = ' '.join(tokens).lower()
                        if cleaned.strip():
                            code_seqs.append(cleaned)

                # # 判断失败条件：除 others 外，其他4个类别都为空
                # non_others_nonempty = any(
                #     len(parsed_result.get(k, [])) > 0
                #     for k in ['function_def', 'loops', 'conditionals', 'assignments']
                # )
                # if not non_others_nonempty:
                #     no_ex_seqs_num += 1
            else:
                # grammer_result = split_broken_julia_structure(raw_code)
                # parsed_result = grammer_result
                #
                # for key in ['function_def', 'loops', 'conditionals', 'assignments', 'others']:
                #     for stmt in grammer_result.get(key, []):
                #         tokens = stmt.split()
                #         cleaned = ' '.join(tokens).lower()
                #         if cleaned.strip():
                #             code_seqs.append(cleaned)
                ast_failed_num += 1

            out_js = {
                'idx': idx,
                'raw_codes': raw_code,
                'cleaned_seqs': code_seqs,
                'function_def': parsed_result.get('function_def', []),
                'loops': parsed_result.get('loops', []),
                'conditionals': parsed_result.get('conditionals', []),
                'assignments': parsed_result.get('assignments', []),
                'others': parsed_result.get('others', [])
            }
            if len(out_js.get("cleaned_seqs")) == len(out_js.get("function_def")) and len(
                    out_js.get("cleaned_seqs")) != 0:
                only_fuction_def += 1
            if len(out_js.get("cleaned_seqs")) == 0:
                no_ex_seqs_num += 1

            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    print('AST parse failed num:', ast_failed_num)
    print('only_fuction_def num:', only_fuction_def)
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
        print('AST parse failed %:', np.round(ast_failed_num / total_num, 4))
        print('only_fuction_def num %:', np.round(only_fuction_def / total_num, 4))
    else:
        print('no ex seqs %: 0.0')
        print('AST parse failed %: 0.0')


def make_LRPL_sentences_structure_label(nl_txt_path, input_jsonl_path, output_jsonl_path):
    """
    专为LRPL，处理真实label，以供和pred出的label，做acc准确度计算
    """
    # 1. 读取 index:summary 映射
    idx_to_nl = {}
    with open(nl_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                idx, summary = line.split(':', 1)
                idx_to_nl[idx.strip()] = summary.strip()

    total_num = 0
    no_ex_seqs_num = 0

    # 2. 打开输入和输出文件
    with jsonlines.open(input_jsonl_path, 'r') as in_f, \
            jsonlines.open(output_jsonl_path, 'w') as out_f:

        for item in tqdm(in_f, desc="Processing and labeling LRPL"):
            idx = item['idx']
            cleaned_seqs = item.get('cleaned_seqs', [])
            nl_text = idx_to_nl.get(idx, "")
            nl_seq = [nl_text.lower()] if nl_text else []

            # 3. 调用 get_code_ex 获取标签和指标
            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(cleaned_seqs, nl_seq)

            # 4. 构造标签
            if len(ex_seqs) == 0:
                ex_seqs = [' '.join(' '.join(cleaned_seqs).split())] if cleaned_seqs else ['']
                no_ex_seqs_num += 1
                ex_labels = [0] * len(cleaned_seqs) if cleaned_seqs else [0]
            else:
                ex_labels = [1 if i in ex_ids else 0 for i in range(len(cleaned_seqs))]

            # 5. 构造输出 item
            out_item = {
                'idx': idx,
                'raw_code': item.get('raw_codes', ''),
                'cleaned_seqs': cleaned_seqs,
                'ex_labels': ex_labels,
                'cleaned_seqs_ex': ex_seqs,
                'nl': nl_seq,
                'fs': fs,
                'ps': ps,
                'rs': rs,
                'max_Rouge_l_r': max_Rouge_l_r,
                'function_def': item.get('function_def', []),
                'loops': item.get('loops', []),
                'conditionals': item.get('conditionals', []),
                'assignments': item.get('assignments', []),
                'others': item.get('others', [])
            }

            out_f.write(out_item)
            total_num += 1

    print('Total processed:', total_num)
    print('No ex seqs num:', no_ex_seqs_num)
    if total_num > 0:
        print('No ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
    else:
        print('No ex seqs %: 0.0')

#允许多个def
def make_LRPL_sentences_structure_new(input_path, output_path, stop_index, r_script_path):
    total_num = 0
    no_ex_seqs_num = 0
    ast_failed_num = 0
    only_fuction_def = 0
    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for line in tqdm(in_f, total=total_lines, desc=f"Processing LRPL {os.path.basename(input_path)}"):

            data = json.loads(line.strip())
            idx = data['index']
            # --- START: 修复代码 ---
            # 1. 读取包含 "\\n" 的单行代码
            raw_code_line = data['best_python_code']

            # 2. 将 "\\n" (两个字符) 替换回 "\n" (一个换行符)
            # --- START: 修复代码 ---
            # try:
            #     # 1. 尝试解码
            #     raw_code = raw_code_line.encode('latin-1').decode('unicode_escape')
            # except Exception:
            #     # 兜底方案
            #     raw_code = raw_code_line.replace('\\n', '\n').replace('\\t', '\t')
            #
            # # 2. 移除 Null Bytes (之前的修复)
            # if '\x00' in raw_code:
            #     raw_code = raw_code.replace('\x00', '')
            #
            # # 3. 【新增关键修复】移除代理字符 (Surrogates)
            # # 这行代码会将无法编码的非法字符直接丢弃 ('ignore')，防止 AST 报错
            # raw_code = raw_code.encode('utf-8', 'ignore').decode('utf-8')

            raw_code = raw_code_line.strip()

            # --- END: 修复代码 ---
            if int(idx) == stop_index:
                break

            parsed_result, ast_success = split_python_by_structure_new(raw_code)

            code_seqs = []
            if ast_success:
                # --- 修改开始：统一处理所有列表类型的键 ---

                # 原来的 'function_def' 是字符串，现在改成了列表，
                # 所以它可以和 'loops', 'conditionals' 等一样处理了！

                target_keys = ['function_def', 'loops', 'conditionals', 'assignments', 'others']

                for key in target_keys:
                    # 获取列表 (比如多个 def 的签名，或多个循环)
                    stmts = parsed_result.get(key, [])

                    # 容错：如果是字符串(旧代码残留)转为列表，如果是列表则保持
                    if isinstance(stmts, str):
                        stmts = [stmts]

                    for stmt in stmts:
                        if not stmt: continue  # 跳过空内容

                        tokens = stmt.split()
                        cleaned = ' '.join(tokens).lower()
                        if cleaned.strip():
                            code_seqs.append(cleaned)

                # --- 修改结束 ---
            else:
                # grammer_result = split_broken_julia_structure(raw_code)
                # parsed_result = grammer_result
                #
                # for key in ['function_def', 'loops', 'conditionals', 'assignments', 'others']:
                #     for stmt in grammer_result.get(key, []):
                #         tokens = stmt.split()
                #         cleaned = ' '.join(tokens).lower()
                #         if cleaned.strip():
                #             code_seqs.append(cleaned)
                ast_failed_num += 1

            out_js = {
                'idx': idx,
                'raw_codes': raw_code,
                'cleaned_seqs': code_seqs,
                'function_def': parsed_result.get('function_def', []),
                'loops': parsed_result.get('loops', []),
                'conditionals': parsed_result.get('conditionals', []),
                'assignments': parsed_result.get('assignments', []),
                'others': parsed_result.get('others', [])
            }
            if len(out_js.get("cleaned_seqs")) == len(out_js.get("function_def")) and len(
                    out_js.get("cleaned_seqs")) != 0:
                only_fuction_def += 1
            if len(out_js.get("cleaned_seqs")) == 0:
                no_ex_seqs_num += 1

            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    print('AST parse failed num:', ast_failed_num)
    print('only_fuction_def num:', only_fuction_def)
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
        print('AST parse failed %:', np.round(ast_failed_num / total_num, 4))
        print('only_fuction_def num %:', np.round(only_fuction_def / total_num, 4))
    else:
        print('no ex seqs %: 0.0')
        print('AST parse failed %: 0.0')

#这是为了和ASAP对比的
def make_retrieved_sentences_structure(input_path, output_path, stop_index):
    def _process_single_code(raw_code):
        """
        辅助函数：处理单个代码片段，返回结构化信息和统计状态
        """
        if not raw_code:
            return {}, False, True  # result, ast_success, is_empty

        # 调用你的AST切分函数
        parsed_result, ast_success = split_ruby_by_structure(raw_code)

        code_seqs = []

        if ast_success:
            # 1. 处理 function_def
            func_def_str = parsed_result.get('function_def', '')
            if func_def_str and func_def_str.strip():
                tokens = func_def_str.split()
                cleaned = ' '.join(tokens).lower()
                if cleaned.strip():
                    code_seqs.append(cleaned)

            # 2. 处理其他键
            for key in ['loops', 'conditionals', 'assignments', 'others']:
                for stmt in parsed_result.get(key, []):
                    tokens = stmt.split()
                    cleaned = ' '.join(tokens).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

        # 构造返回的结构部分
        structure_info = {
            'cleaned_seqs': code_seqs,
            'function_def': parsed_result.get('function_def', []) if ast_success else [],
            'loops': parsed_result.get('loops', []) if ast_success else [],
            'conditionals': parsed_result.get('conditionals', []) if ast_success else [],
            'assignments': parsed_result.get('assignments', []) if ast_success else [],
            'others': parsed_result.get('others', []) if ast_success else []
        }

        return structure_info, ast_success, False
    total_num = 0

    # 统计主要针对外层 Main Code 的表现
    main_no_ex_seqs_num = 0
    main_ast_failed_num = 0
    main_only_function_def = 0

    # 简单的统计一下 retrieved 的情况
    retrieved_ast_failed_total = 0

    # 计算总行数用于 tqdm
    total_lines = 0
    with open(input_path, 'r', encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, 'r', encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for i, line in tqdm(enumerate(in_f), total=total_lines, desc=f"Processing {os.path.basename(input_path)}"):

            if stop_index is not None and i == stop_index:
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i}, json decode error.")
                continue

            # ==========================================
            # 1. 处理外层代码 (Main Code)
            # ==========================================
            raw_code = data.get("code", "")

            # 这里不再需要 split('\t') 和 replace('\\n')，因为 JSON load 应该已经自动处理了转义
            # 如果你的 JSON 里面的 code 仍然是二次转义的，请保留原来的 decode('unicode_escape') 逻辑
            # raw_code = raw_code.encode('latin-1').decode('unicode_escape')

            structure_info, ast_success, is_empty = _process_single_code(raw_code)

            # 更新外层字典
            data.update(structure_info)

            # 统计外层代码指标
            if not ast_success:
                main_ast_failed_num += 1

            if len(data.get("cleaned_seqs", [])) == 0:
                main_no_ex_seqs_num += 1

            if ast_success and len(data.get("cleaned_seqs", [])) == len(data.get("function_def", [])) and len(
                    data.get("cleaned_seqs", [])) != 0:
                main_only_function_def += 1

            # ==========================================
            # 2. 处理 retrieved_candidates (3个示例)
            # ==========================================
            candidates = data.get("retrieved_candidates", [])
            for cand in candidates:
                cand_code = cand.get("code", "")

                # 处理单个 candidate
                cand_struct, cand_ast_success, _ = _process_single_code(cand_code)

                # 将结构信息更新到 candidate 自己的字典里
                cand.update(cand_struct)

                if not cand_ast_success:
                    retrieved_ast_failed_total += 1

            # ==========================================
            # 3. 写入文件
            # ==========================================
            out_f.write(data)
            total_num += 1

    print('=' * 30)
    print(f'Total processed: {total_num}')
    print('--- Main Code Stats ---')
    print(f'AST parse failed num: {main_ast_failed_num}')
    print(f'No ex seqs num: {main_no_ex_seqs_num}')
    print(f'Only function_def num: {main_only_function_def}')

    print('--- Retrieved Code Stats ---')
    print(f'Total Retrieved AST failed events: {retrieved_ast_failed_total}')

    if total_num > 0:
        print('-' * 20)
        print('Main AST parse failed %:', np.round(main_ast_failed_num / total_num, 4))
        print('Main no ex seqs %:', np.round(main_no_ex_seqs_num / total_num, 4))
        print('Main only_function_def %:', np.round(main_only_function_def / total_num, 4))
    else:
        print('No data processed.')

#这是针对我方法，输入是翻译的python
def make_retrieved_sentences_structure_final_result(input_path, output_path, stop_index):
    def _process_single_code(raw_code):
        """
        辅助函数：处理单个代码片段，返回结构化信息和统计状态
        """
        if not raw_code:
            return {}, False, True  # result, ast_success, is_empty

        # --- 修改点：调用新的 AST 切分函数 ---
        parsed_result, ast_success = split_python_by_structure_new(raw_code)

        code_seqs = []

        if ast_success:
            # 1. 处理 function_def (现在是一个 list)
            # --- 修改点：遍历 list ---
            for func_sig in parsed_result.get('function_def', []):
                if func_sig and func_sig.strip():
                    tokens = func_sig.split()
                    cleaned = ' '.join(tokens).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

            # 2. 处理其他键 (这些本来就是 list，逻辑不变，但保持统一)
            for key in ['loops', 'conditionals', 'assignments', 'others']:
                for stmt in parsed_result.get(key, []):
                    tokens = stmt.split()
                    cleaned = ' '.join(tokens).lower()
                    if cleaned.strip():
                        code_seqs.append(cleaned)

        # 构造返回的结构部分
        structure_info = {
            'cleaned_seqs': code_seqs,
            'function_def': parsed_result.get('function_def', []) if ast_success else [],
            'loops': parsed_result.get('loops', []) if ast_success else [],
            'conditionals': parsed_result.get('conditionals', []) if ast_success else [],
            'assignments': parsed_result.get('assignments', []) if ast_success else [],
            'others': parsed_result.get('others', []) if ast_success else []
        }

        return structure_info, ast_success, False

    total_num = 0

    # 统计主要针对外层 Main Code 的表现
    main_no_ex_seqs_num = 0
    main_ast_failed_num = 0
    main_only_function_def = 0

    # 简单的统计一下 retrieved 的情况
    retrieved_ast_failed_total = 0

    # 计算总行数用于 tqdm
    total_lines = 0
    try:
        with open(input_path, 'r', encoding="utf-8") as f:
            for _ in f:
                total_lines += 1
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        for i, line in tqdm(enumerate(in_f), total=total_lines, desc=f"Processing {os.path.basename(input_path)}"):

            if stop_index is not None and i == stop_index:
                break

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {i}, json decode error.")
                continue

            # ==========================================
            # 1. 处理外层代码 (Main Code)
            # ==========================================

            # --- 关键修正：读取翻译后的 Python 代码，而非原 LRPL 代码 ---
            # 优先读取 best_python_code，其次 python_code
            raw_code = data.get("best_python_code", "")
            if not raw_code:
                raw_code = data.get("python_code", "")

            # 如果依然为空，就不读取 "code" (因为那是 LRPL)，直接视为空代码处理
            # 这样可以避免 AST Parse Failed 刷屏

            structure_info, ast_success, is_empty = _process_single_code(raw_code)

            # 更新外层字典
            data.update(structure_info)

            # 统计外层代码指标
            if not ast_success and not is_empty:  # 只有非空且解析失败才算 Fail
                main_ast_failed_num += 1

            if len(data.get("cleaned_seqs", [])) == 0:
                main_no_ex_seqs_num += 1

            # 这里的逻辑依然成立：如果 cleaned_seqs 的数量等于 function_def 的数量
            # 说明除了函数签名外，没有提取到其他有效 body（loops/conditionals 等）
            if ast_success and len(data.get("cleaned_seqs", [])) == len(data.get("function_def", [])) and len(
                    data.get("cleaned_seqs", [])) != 0:
                main_only_function_def += 1

            # ==========================================
            # 2. 处理 retrieved_candidates (3个示例)
            # ==========================================
            candidates = data.get("retrieved_candidates", [])
            for cand in candidates:
                # 检索库里的代码本身就是 Python，通常存在 'code' 字段里，这里不需要改
                cand_code = cand.get("code", "")

                # 处理单个 candidate
                cand_struct, cand_ast_success, _ = _process_single_code(cand_code)

                # 将结构信息更新到 candidate 自己的字典里
                cand.update(cand_struct)

                if not cand_ast_success and cand_code:
                    retrieved_ast_failed_total += 1

            # ==========================================
            # 3. 写入文件
            # ==========================================
            out_f.write(data)
            total_num += 1

    print('=' * 30)
    print(f'Total processed: {total_num}')
    print('--- Main Code Stats (Translated Python) ---')
    print(f'AST parse failed num: {main_ast_failed_num}')
    print(f'No ex seqs num: {main_no_ex_seqs_num}')
    print(f'Only function_def num: {main_only_function_def}')

    print('--- Retrieved Code Stats ---')
    print(f'Total Retrieved AST failed events: {retrieved_ast_failed_total}')

    if total_num > 0:
        print('-' * 20)
        print('Main AST parse failed %:', np.round(main_ast_failed_num / total_num, 4))
        print('Main no ex seqs %:', np.round(main_no_ex_seqs_num / total_num, 4))
        print('Main only_function_def %:', np.round(main_only_function_def / total_num, 4))
    else:
        print('No data processed.')

if __name__ == '__main__':
    # langs = ['julia', 'lua', 'ocaml', 'r', 'racket']
    langs=['ruby']
    language = 'python'  # <--- 确保这里是你想要的语言
    for lang in langs:
        make_retrieved_sentences_structure(
            f'../../../../dataset/CSN/ruby/test.jsonl',
            f'../../../../dataset/CSN/ruby/trans_qwen3th/test_sentences.jsonl',
            3759
        )
    # make_LRPL_sentences_structure_label(
    # '../../../../experiment/ds-coder-1_3B/lua_result/lua_ref_48194.txt',
    # '../../../../dataset/LowData/lua/Lua_structure_all.jsonl',
    # '../../../../dataset/LowData/lua/Lua_structure_real_labels_all.jsonl'
    # )
    # 确保路径是正确的
    # input_root = f'../../../../dataset/CSN/{language}/'
    # output_root = f'../../../../dataset/CSN/{language}-cls/'


    #
    # make_pcsd_dataset_structure(input_root, output_root,'python')
    # 确保输出目录存在
    # if not os.path.exists(output_root):
    #     os.makedirs(output_root)

    dataset_file = ['train', 'valid', 'test']
    # t = 0
    # f1 = 0
    # with open("../../../../experiment/ds-coder-1_3B/r_result/r_2_python_clean.txt", 'r', encoding='utf-8') as f:
    #     for line in f:
    #         code = line.split(':')[1].strip()
    #
    #         try:
    #             ast.parse(code)
    #             t += 1
    #             print("True")
    #         except Exception as e:
    #             f1 += 1
    #             print("False")
    #     print("t",t)
    #     print("f",f1)

    # langs=['r','racket']
    # for lang in langs:
    #     capitalized_lang = lang.capitalize()
    #     input_root = f'../../../../dataset/LowData/{lang}/{lang}_python_best_candidate_vllm_B0.5_S0.5.jsonl'
    #     output_root = f'../../../../dataset/LowData/{lang}/{lang}_structure_B0.5_S0.5.jsonl'
    #     make_LRPL_sentences_structure_new(input_root, output_root, -1,'')
    # make_LRPL_sentences_structure(input_root, output_root, -1, "../../../../vendor/parse_structure.R")

    # for i in dataset_file:
    #     input_path = input_root + i + "/" + i + '.jsonl'
    #     output_path = output_root + i + "/" + i + '.jsonl'
    #     # 检查输入文件是否存在
    #
    #     if os.path.exists(input_path):
    #         make_pcsd_dataset_structure(input_path, output_path, language)
    #     else:
    #         print(f"Input file not found: {input_path}")
