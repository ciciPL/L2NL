import json
import os  # <--- 添加导入 os 库
from vendor import parse_R
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
    id_sort_by_scores = np.argsort(scores)[::-1]
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

def make_LRPL_sentences_ast(input_path, output_path,stop_index):
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
            code_statement_strings, ast_success = parse_R.split_r_into_statements(raw_code,'../../../../vendor/parse_fixed.R')  # <--- 接收标志

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

            if len(code_statement_strings)==1:
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

if __name__ == '__main__':
    language = 'python'  # <--- 确保这里是你想要的语言

    # 确保路径是正确的
    # input_root = f'../../../../dataset/CSN/{language}/'
    # output_root = f'../../../../dataset/CSN/{language}-cls/'

    input_root = f'../../../../dataset/finetune/aliPCSDData/output6k.json'
    output_root = f'../../../../dataset/ready_sentences_dataset/6k_sentences.json'

    make_pcsd_dataset(input_root, output_root,language)
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

    # make_LRPL_sentences_ast(input_root,output_root,3760)

    # for i in dataset_file:
    #     input_path = input_root + i + "/" + i + '.jsonl'
    #     output_path = output_root + i + "/" + i + '.jsonl'
    #     # 检查输入文件是否存在
    #
    #     if os.path.exists(input_path):
    #         make_pcsd_dataset(input_path, output_path, language)
    #     else:
    #         print(f"Input file not found: {input_path}")
