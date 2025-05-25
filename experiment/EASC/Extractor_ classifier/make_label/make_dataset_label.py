import json
import jsonlines
import numpy as np
from tqdm import tqdm
import rouge_not_a_wrapper as my_rouge
from identifier_splitting import split_identifier_into_parts
import os  # <--- 添加导入 os 库

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

    # 计算总行数以完善 tqdm
    total_lines = 0
    with open(input_path, encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    with open(input_path, encoding="utf-8") as in_f, jsonlines.open(output_path, mode='w') as out_f:
        # 使用 tqdm 并传入总行数
        for line in tqdm(in_f, total=total_lines, desc=f"Processing PCSD {os.path.basename(input_path)}"):
            line = line.strip()
            js = json.loads(line)

            # 使用 PCSD 数据集的字段
            idx = js['id']
            # 假设 'code' 字段是已经分词但未分割标识符的字符串
            code_tokens = js['code'].split()
            # 假设 'comment' 字段是已经分词但未分割标识符的字符串
            nl_tokens = js['comment'].split()

            # --- 后续处理与 make_py_dataset 类似 ---

            # 1. 分割代码 (使用现有的 split_java_to_seqs，注意其对 Python 的局限性)
            code_seqs = split_python_to_seqs(code_tokens)
            code_seqs = ' <spt> '.join(code_seqs)

            # 2. 分割标识符并小写
            code_seqs = split_identifier_into_parts(code_seqs)
            code_seqs = ' '.join(code_seqs)
            code_seqs_lower = code_seqs.lower()
            code_seqs = code_seqs_lower.split(' <spt> ')
            code_seqs = [seq for seq in code_seqs if seq.strip()]  # 过滤空序列

            # 3. 处理完整代码（用于回退）
            code = ' '.join(code_tokens).replace('\n', '')
            code = split_identifier_into_parts(code)
            code_lower = [s.lower() for s in code]

            # 4. 处理 NL
            nl = ' '.join(nl_tokens).replace('\n', '')
            nl = split_identifier_into_parts(nl)
            nl_seq = [' '.join(nl).lower()]

            # 5. 准备输出 JSON 对象
            out_js = {}
            out_js['idx'] = idx
            out_js['cleaned_nl'] = nl_seq
            out_js['cleaned_codes'] = code_lower
            out_js['cleaned_blocks'] = []  # 保持字段存在，但为空
            out_js['cleaned_blocks_ex'] = []  # 保持字段存在，但为空

            # 6. 获取重要序列和标签
            ex_seqs, ex_ids, fs, ps, rs, max_Rouge_l_r = get_code_ex(code_seqs, nl_seq)
            if len(ex_seqs) == 0:
                ex_seqs = [' '.join(code_lower)]  # 回退策略
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
            # 确保标签存在
            if 'ex_labels' not in out_js:
                out_js['ex_labels'] = [1 if i in ex_ids else 0 for i in range(len(code_seqs))]

            # 7. 写入文件
            out_f.write(out_js)
            total_num += 1

    print('total num:', total_num)
    print('no ex seqs num:', no_ex_seqs_num)
    if total_num > 0:
        print('no ex seqs %:', np.round(no_ex_seqs_num / total_num, 4))
    else:
        print('no ex seqs %: 0.0')


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


if __name__ == '__main__':
    language = 'python'  # <--- 确保这里是你想要的语言

    # 确保路径是正确的
    # input_root = f'../../../../dataset/CSN/{language}/'
    # output_root = f'../../../../dataset/CSN/{language}-cls/'

    input_root = f'../../../../finetune/dataset/Clean_PCSD/'
    output_root = f'../../../../finetune/dataset/Clean_PCSD-cls/'
    # 确保输出目录存在
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    dataset_file = ['train', 'valid', 'test']

    for i in dataset_file:
        input_path = input_root + i + "/" + i + '.jsonl'
        output_path = output_root + i + "/" + i + '.jsonl'
        # 检查输入文件是否存在
        if os.path.exists(input_path):
            make_pcsd_dataset(input_path, output_path, language)
        else:
            print(f"Input file not found: {input_path}")
