import ast
import json
import os
import subprocess
import tokenize
from io import StringIO
import sys

import tree_sitter_julia
import tree_sitter_ocaml
from tree_sitter import Language, Parser


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
        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs

# def split_python_to_seqs(code_seq, code_token_list):
#     '''
#     1. 需要去除\'''\'''的内容
#     2. 分句
#     3. 分tokens
#         1. 需要加空格的token ( ) : + - = . , * / += -= *= /= //
#         2. 但是...不分
#         3. ''和""里面的内容不分
#     '''
#     seq_lines = code_seq.splitlines()
#     seq_lines = [i.strip() for i in seq_lines]
#     seq_lines = [i for i in seq_lines if len(i) > 0]
#
#     comment_1 = False
#     comment_2 = False
#
#     comment_1_in = False
#     comment_2_in = False
#
#     cleaned_seq_lines = []
#
#     for i in seq_lines:
#         if i.startswith('#'):
#             continue
#
#         if not comment_1 and i.startswith(r'"""'):
#             comment_1 = True
#             continue
#         if comment_1 and i.startswith(r'"""'):
#             comment_1 = False
#             continue
#
#         if not comment_2 and i.startswith(r"'''"):
#             comment_2 = True
#             continue
#         if comment_2 and i.startswith(r"'''"):
#             comment_2 = False
#             continue
#
#         if comment_1 or comment_2:
#             continue
#
#         if r"'''" in i and not i.startswith(r"'''"):
#             if i.count(r"'''") == 2:
#                 s_idx = i.find(r"'''")
#                 e_idx = len(i) - i[::-1].find(r"'''")
#                 cleaned_seq = i[:s_idx] + i[e_idx:]
#                 cleaned_seq_lines.append(cleaned_seq)
#             elif i.count(r"'''") == 1 and not comment_1_in:
#                 s_idx = i.find(r"'''")
#                 cleaned_seq_lines.append(i[:s_idx])
#                 comment_1_in = True
#             # elif:
#             #     pass
#
#         if r'"""' in i:
#             pass
#
#         if comment_1_in or comment_2_in:
#             continue
#
#         seq = i
#         seq_temp = ''
#         while True:
#             if '#' in seq:
#                 quo_1 = 0
#                 quo_2 = 0
#                 last_index = seq[::-1].find('#')
#                 for j in seq[:last_index]:
#                     if j == "\'":
#                         quo_1 += 1
#
#                     if j == '\"':
#                         quo_2 += 1
#
#                 if quo_1 % 2 == 0 and quo_2 % 2 == 0:
#                     seq_temp += seq[:last_index]
#                     seq = seq_temp
#                 else:
#                     break
#             else:
#                 break
#
#         cleaned_seq_lines.append(seq)
#
#     seq_idx = 0
#     token_start_idx = 0
#     token_end_idx = 0
#
#     code_seqs = []
#     # for cl in cleaned_seq_lines:
#     #     new_cl = cl.replace('(',' ( ').replace(')',' ) ').replace('{',' { ').replace('(',' ( ')
#     #     code_seqs.append(new_cl)
#
#     for idx, cl in enumerate(code_token_list):
#         if seq_idx == len(cleaned_seq_lines) - 1:
#             code_seqs.append(' '.join(code_token_list[token_end_idx + 1:]))
#             break
#
#         if cleaned_seq_lines[seq_idx].startswith(cl):
#             token_start_idx = idx
#             continue
#         if cleaned_seq_lines[seq_idx].endswith(cl):
#             if seq_idx < len(cleaned_seq_lines) - 1 and idx < len(code_token_list) - 1:
#                 if cleaned_seq_lines[seq_idx + 1].startswith(code_token_list[idx + 1]):
#                     token_end_idx = idx
#                     code_seqs.append(' '.join(code_token_list[token_start_idx:token_end_idx + 1]))
#                     seq_idx += 1
#                 else:
#                     continue
#             else:
#                 token_end_idx = idx
#                 code_seqs.append(' '.join(code_token_list[token_start_idx:token_end_idx + 1]))
#                 seq_idx += 1
#
#     if seq_idx != len(cleaned_seq_lines) - 1:
#         print('aa')
#
#     return code_seqs
def get_tokens(code_string):
    """Helper to tokenize a code string."""
    tokens = []
    try:
        # Use tokenize for more robust tokenization than split()
        for tok in tokenize.generate_tokens(StringIO(code_string).readline):
            if tok.type not in [tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE]:
                 # Skip comments, encoding, newlines etc. if needed
                 # or handle them based on how CSN data is tokenized
                if tok.type != tokenize.COMMENT:
                    tokens.append(tok.string)
    except tokenize.TokenError:
         # Fallback if tokenize fails (e.g., incomplete code)
        tokens = code_string.split()
    return tokens

def split_python_with_ast(raw_code):
    """
    Splits Python code using AST.
    Returns a list of code statement strings AND a boolean indicating success.
    """
    sequences = []
    success = True  # 假设成功
    try:
        tree = ast.parse(raw_code)

        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            func_body = tree.body[0].body
        else:
            func_body = tree.body

        for node in func_body:
            try:
                segment = ast.get_source_segment(raw_code, node)
                if segment:
                    sequences.append(segment)
            except AttributeError:
                start_line = node.lineno - 1
                end_line = getattr(node, 'end_lineno', start_line)
                lines = raw_code.splitlines()[start_line:end_line]
                sequences.append("\n".join(lines))

    except SyntaxError:
        print(f"Warning: AST parsing failed for code snippet. Falling back to newline split.")
        sequences = [line.strip() for line in raw_code.strip().split('\n') if line.strip()]
        success = False  # 标记为失败

    return sequences if sequences else [raw_code.strip()], success  # 返回列表和成功标志

def split_python_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # {
    left_brace = 0

    # [
    left_bracket = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=',
        '.', ',', ':',
        '(', '{', '[',
        'and', 'or', 'not', 'is', 'in', 'as'
    ]

    end_idx = 0
    idxs = []

    sp_word = ['return',
               'while', 'for',
               'if', 'elif', 'else',
               'try', 'except', 'Exception', 'raise',
               'assert',
               'yield',
               'with', 'print', ]

    return_sent = 0
    for_sent = 0
    if_sent = 0
    try_sent = 0
    assert_sent = 0
    with_sent = 0
    print_sent = 0
    yield_sent = 0

    def_class_sent = False
    sp_sent = 0

    for idx, i in enumerate(code_token_list):
        if i == 'def' or i == 'class':
            def_class_sent = True

        if i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'if' or i == 'elif' or i == 'else':
            if_sent += 1
        elif i == 'try' or i == 'except' or i == 'Exception' or i == 'raise':
            try_sent += 1
        elif i == 'assert':
            assert_sent += 1
        elif i == 'with':
            with_sent += 1
        elif i == 'print':
            print_sent += 1
        elif i == 'yield':
            yield_sent += 1

        left_up = left_s_quo + left_d_quo + left_par + left_brace + left_bracket
        if i == ':' and left_up == 0:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            def_class_sent = False
            continue
        elif i == r"'" and left_up == 0:
            left_s_quo += 1
        elif i == r'"':
            left_d_quo += 1
        elif i == '(':
            left_par += 1
        elif i == '{':
            left_brace += 1
        elif i == '[':
            left_bracket += 1
        elif i == r"'" and left_s_quo == 1:
            left_s_quo -= 1
        elif i == r'"' and left_d_quo == 1:
            left_d_quo -= 1
        elif i == ')':
            left_par -= 1
        elif i == '}':
            left_brace -= 1
        elif i == ']':
            left_bracket -= 1

        new_left_up = left_s_quo + left_d_quo + left_par + left_brace + left_bracket
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if if_sent == 1:
                    if_sent += 1
                    continue
                elif if_sent == 2:
                    if_sent = 0

                if try_sent == 1:
                    try_sent += 1
                    continue
                elif try_sent == 2:
                    try_sent = 0

                if assert_sent == 1:
                    assert_sent += 1
                    continue
                elif assert_sent == 2:
                    assert_sent = 0

                if with_sent == 1:
                    with_sent += 1
                    continue
                elif with_sent == 2:
                    with_sent = 0

                if print_sent == 1:
                    print_sent += 1
                    continue
                elif print_sent == 2:
                    print_sent = 0

                if yield_sent == 1:
                    yield_sent += 1
                    continue
                elif yield_sent == 2:
                    yield_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not def_class_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ']', ']'] \
                        and not def_class_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        code_seqs.append(' '.join(code_snap))
    return code_seqs

def split_python_by_structure(raw_code):
    try:
        tree = ast.parse(raw_code)
        result = {
            'function_def': '',
            'loops': [],
            'conditionals': [],
            'assignments': [],
            'others': []
        }

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                result['function_def'] = ast.get_source_segment(raw_code, node).split('\n')[0]
                body = node.body
            else:
                body = [node]

            for stmt in body:
                try:
                    segment = ast.get_source_segment(raw_code, stmt)
                except:
                    start = stmt.lineno - 1
                    end = getattr(stmt, 'end_lineno', stmt.lineno)
                    lines = raw_code.splitlines()[start:end]
                    segment = "\n".join(lines)

                if isinstance(stmt, (ast.For, ast.While)):
                    result['loops'].append(segment)
                elif isinstance(stmt, ast.If):
                    # ✅ 仅添加整体 if 块，不展开细节
                    result['conditionals'].append(segment)
                elif isinstance(stmt, ast.Assign):
                    # ✅ 只收集最外层的变量定义
                    result['assignments'].append(segment)
                else:
                    result['others'].append(segment)

        return result, True

    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return {
            'function_def': '',
            'loops': [],
            'conditionals': [],
            'assignments': [],
            'others': []
        }, False

def split_go_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # [
    left_bracket = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=', ':=',
        '.', ',', ':', ';',
        '(', '{', '[',
        'range',
    ]

    func_sent = False
    return_sent = 0
    for_sent = 0
    defer_sent = 0
    var_sent = 0
    switch_sent = 0

    end_idx = 0
    idxs = []

    for idx, i in enumerate(code_token_list):
        left_up = left_par + left_bracket
        if i == '{' and left_up == 0:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            func_sent = False
            continue
        elif i == '}':
            end_idx = idx + 1
            continue
        elif i == '(':
            left_par += 1
        elif i == ')':
            left_par -= 1
        elif i == '[':
            left_bracket += 1
        elif i == ']':
            left_bracket -= 1

        if i == 'func':
            func_sent = True
        elif i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'defer':
            defer_sent += 1
        elif i == 'var':
            var_sent += 1
        elif i == 'switch':
            switch_sent += 1

        new_left_up = left_par + left_bracket
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if defer_sent == 1:
                    defer_sent += 1
                    continue
                elif defer_sent == 2:
                    defer_sent = 0

                if var_sent == 1:
                    var_sent += 1
                    continue
                elif var_sent == 2:
                    var_sent = 0

                if switch_sent == 1:
                    switch_sent += 1
                    continue
                elif switch_sent == 2:
                    switch_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not func_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ')', ']'] \
                        and not func_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]

        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))

    return code_seqs

def split_ruby_to_seqs(code_token_list):
    # '"
    left_s_quo, left_d_quo = 0, 0

    # (
    left_par = 0

    # [
    left_bracket = 0

    # {
    left_brace = 0

    s_word = [
        '=', '+', '-', '*', '/', '%', '^', '~', '!', '|', '&', '<', '>',
        '==', '+=', '-=', '*=', '/=', '%=', '^=', '~=', '!=', '||', '&&',
        '<=', '>=', ':=', '<<', '>>',
        '.', ',', ':', ';', '::',
        '(', '{', '[',
        'do'
    ]

    def_sent = False
    return_sent = 0
    for_sent = 0
    if_sent = 0

    end_idx = 0
    idxs = []

    for idx, i in enumerate(code_token_list):
        if i == 'def':
            def_sent = True
            continue
        if i == ')':
            left_par -= 1
        left_up = left_par + left_bracket + left_brace
        if i == ')' and left_up == 0 and def_sent:
            idxs.append((end_idx, idx))
            end_idx = idx + 1
            def_sent = False
            continue
        elif i == 'end':
            end_idx = idx + 1
            continue
        elif i == '(':
            left_par += 1
        elif i == '[':
            left_bracket += 1
        elif i == ']':
            left_bracket -= 1
        elif i == '{':
            left_brace += 1
        elif i == '}':
            left_brace -= 1

        if i == 'def':
            def_sent = True
        elif i == 'return':
            return_sent += 1
        elif i == 'for' or i == 'while':
            for_sent += 1
        elif i == 'if':
            if_sent += 1

        if idx == 2 and i != '(' and def_sent:
            idxs.append((end_idx, idx))
            end_idx = idx
            def_sent = False
            continue

        new_left_up = left_par + left_bracket + left_brace
        if new_left_up == 0:
            if idx < len(code_token_list) - 1:
                if return_sent == 1:
                    return_sent += 1
                    continue
                elif return_sent == 2:
                    return_sent = 0

                if for_sent == 1:
                    for_sent += 1
                    continue
                elif for_sent == 2:
                    for_sent = 0

                if if_sent == 1:
                    if_sent += 1
                    continue
                elif if_sent == 2:
                    if_sent = 0

                if code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] not in s_word \
                        and not def_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
                elif code_token_list[idx + 1] not in s_word \
                        and code_token_list[idx] in ['}', ')', ']'] \
                        and not def_sent:
                    idxs.append((end_idx, idx))
                    end_idx = idx + 1
            else:
                idxs.append((end_idx, idx))

    code_seqs = []
    for idx, i in enumerate(idxs):
        code_snap = code_token_list[i[0]:i[1] + 1]
        if ' '.join(code_snap) == 'return' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]
        if ' '.join(code_snap) == 'if' and idx < len(idxs) - 1:
            code_snap = code_token_list[i[0]:idxs[idx + 1][1] + 1]

        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))

    return code_seqs

def split_php_to_seqs(code_token_list):
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
        if code_snap[-1] == '{':
            code_seqs.append(' '.join(code_snap) + ' }')
        else:
            code_seqs.append(' '.join(code_snap))
    return code_seqs

def split_r_by_structure(raw_r_code: str, r_script_path: str) -> tuple[dict, bool]:
    """
    使用结构化 tree-sitter-R 语法分析，将 R 代码分为所需的结构类型。
    返回结构字典 + 是否成功。
    """
    default_result = {
        "function_def": [],
        "loops": [],
        "conditionals": [],
        "assignments": [],
        "others": []
    }

    if not raw_r_code.strip():
        return default_result, True

    if not os.path.exists(r_script_path):
        print(f"[错误] R 脚本未找到: {r_script_path}")
        return default_result, False

    try:
        process = subprocess.run(
            ["Rscript", r_script_path, raw_r_code],
            capture_output=True,
            text=True,
            check=True,
            encoding='UTF-8'
        )
        # # 打印 stderr 方便调试 R 代码中的 message() 输出
        # if process.stderr.strip():
        #     print("[R stderr]:")
        #     print(process.stderr.strip())

        output = process.stdout.strip()
        if not output:
            print("[警告] R 脚本没有输出内容")
            return default_result, True

        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as e:
            print("[JSON解析错误] 无法解析 R 脚本输出为 JSON:")
            print(output)
            print(f"错误信息: {e}")
            return default_result, False

        # 确保所有关键字段存在
        for key in default_result:
            if key not in parsed:
                parsed[key] = []

        return parsed, True

    except subprocess.CalledProcessError as e:
        print(f"Rscript 执行失败:\n{e.stderr}")
        return default_result, False
    except Exception as e:
        print(f"其他错误: {e}")
        return default_result, False

def split_ocaml_by_structure(ocaml_code: str):

    # --- 辅助函数 ---
    def node_text(node, code_bytes):
        if not node: return ""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    def find_child_by_type(parent_node, target_type, recursive=False):
        if not parent_node: return None
        queue = list(parent_node.children)
        while queue:
            node = queue.pop(0)
            if node.type == target_type:
                return node
            if recursive:
                queue.extend(node.children)
        return None

    # --- 核心处理函数 ---
    def classify_node(node, code_bytes, results_dict):
        if not node or not node.is_named:
            return

        node_type = node.type

        if node_type == 'value_definition':
            let_binding_node = find_child_by_type(node, 'let_binding')
            if not let_binding_node: return

            is_function = find_child_by_type(let_binding_node, 'parameter') is not None

            if is_function:
                body_node = let_binding_node.child_by_field_name('body')
                full_text = node_text(node, code_bytes)
                sig_text = full_text
                if body_node:
                    body_text = node_text(body_node, code_bytes)
                    sig_end_index = full_text.rfind(body_text)
                    if sig_end_index != -1:
                        cleaned_sig = full_text[:sig_end_index].strip().replace('\n', ' ').replace('  ', ' ')
                        sig_text = cleaned_sig + "= {}"
                results_dict['function_def'].append(sig_text)
                classify_node(body_node, code_bytes, results_dict)
                return

            else:
                results_dict['bindings'].append(node_text(node, code_bytes))
                body_node = let_binding_node.child_by_field_name('body')
                if body_node:
                    classify_node(body_node, code_bytes, results_dict)
                return

        elif node_type in ['for_expression', 'while_expression']:
            results_dict['loops'].append(node_text(node, code_bytes))
            body_node = node.child_by_field_name('body')
            classify_node(body_node, code_bytes, results_dict)
            return

        elif node_type in ['if_expression', 'match_expression']:
            results_dict['conditionals'].append(node_text(node, code_bytes))
            classify_node(node.child_by_field_name('consequence'), code_bytes, results_dict)
            classify_node(node.child_by_field_name('alternative'), code_bytes, results_dict)
            return

        elif node_type == 'let_expression':
            for child in node.children:
                classify_node(child, code_bytes, results_dict)
            return

        elif node_type not in ['compilation_unit', 'comment', 'let', 'rec', '=', 'let_binding', 'then_clause',
                               'else_clause', 'then', 'else', ';', 'in', '(', ')', '{', '}', 'begin', 'end', 'do']:
            results_dict['others'].append(node_text(node, code_bytes))

    # --- 主逻辑 ---
    code_bytes = ocaml_code.encode('utf8')

    try:
        OCAML_LANGUAGE = Language(tree_sitter_ocaml.language_ocaml())
        parser = Parser(OCAML_LANGUAGE)
    except Exception as e:
        error_message = f"无法初始化OCaml解析器: {e}"
        print(error_message)
        return {}, False

    tree = parser.parse(code_bytes)
    root_node = tree.root_node

    if root_node.has_error:
        error_node = find_child_by_type(root_node, 'ERROR', recursive=True) or root_node
        line, col = error_node.start_point
        error_message = f"解析错误: OCaml代码在行 {line + 1}, 列 {col + 1} 附近包含语法错误。"
        print(error_message)
        return {}, False

    results = {
        "function_def": [],
        "loops": [],
        "conditionals": [],
        "bindings": [],
        "others": []
    }
    for top_level_statement in root_node.children:
        classify_node(top_level_statement, code_bytes, results)

    # **修改点**: 返回布尔值和字典，而不是打包后的字典
    return results, True

def split_julia_by_structure(julia_code: str):

    # --- 辅助函数 ---
    def node_text(node, code_bytes):
        if not node: return ""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    # --- 核心处理函数 (已修正) ---
    def classify_node(node, code_bytes, results_dict):
        # 忽略不重要或无名的节点
        if not node or not node.is_named or node.type in ['line_comment', 'block_comment', 'end']:
            return

        node_type = node.type

        # Case 1: 函数定义 (容器，需要递归)
        if node_type == 'function_definition':
            signature_node = next((c for c in node.children if c.type == 'signature'), None)
            sig_text = node_text(signature_node, code_bytes) if signature_node else "function"
            results_dict['function_def'].append(f"function {sig_text} ... end")

            # 递归处理函数体内的所有语句
            in_body = False
            for child in node.children:
                if signature_node and child.id == signature_node.id:
                    in_body = True
                    continue
                if child.type == 'end':
                    break
                if in_body:
                    classify_node(child, code_bytes, results_dict)
            return

        # Case 2: 循环 (最终语句，不再深入)
        elif node_type in ['for_statement', 'while_statement']:
            results_dict['loops'].append(node_text(node, code_bytes))
            return

        # Case 3: 条件判断 (最终语句，不再深入)
        elif node_type == 'if_statement':
            results_dict['conditionals'].append(node_text(node, code_bytes))
            return

        # Case 4: 赋值/绑定 (最终语句，不再深入)
        elif node_type == 'assignment':
            results_dict['bindings'].append(node_text(node, code_bytes))
            return

        # **关键修正**: 显式处理其他类型的独立语句，将它们归入 'others'
        elif node_type in ['return_statement', 'call_expression', 'macro_call']:
            results_dict['others'].append(node_text(node, code_bytes))
            return

        # Case 6: 如果节点本身未被分类 (例如 'source_file', 'block'),
        # 它只是一个结构化容器，则继续深入其子节点。
        for child in node.children:
            classify_node(child, code_bytes, results_dict)

    # --- 主逻辑 ---
    code_bytes = julia_code.encode('utf8')
    try:
        JULIA_LANGUAGE = Language(tree_sitter_julia.language())
        parser = Parser(JULIA_LANGUAGE)
    except Exception as e:
        print(e)

    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    if root_node.has_error:
        return {}, False

    results = {
        "function_def": [], "loops": [], "conditionals": [], "bindings": [], "others": []
    }
    classify_node(root_node, code_bytes, results)
    return  results,True

