import ctypes
import json
import sys
import os
import tree_sitter_ocaml
from tree_sitter import Language, Parser
import tree_sitter_julia as julia_lang
import tree_sitter_lua

def split_julia_by_structure(julia_code: str):
    """
    解析Julia代码字符串，并将其扁平化、互不重叠地分类。

    Args:
        julia_code (str): 要解析的 Julia 源代码。

    Returns:
        tuple[bool, dict | str]: 一个包含两个元素的元组:
              - success (bool): 解析是否成功。
              - result (dict | str): 成功时是包含分类结果的字典，
                                     失败时是包含错误信息的字符串。
    """
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
        JULIA_LANGUAGE = Language(julia_lang.language())
        parser = Parser(JULIA_LANGUAGE)
    except Exception as e:
        return False, f"无法初始化Julia解析器: {e}"

    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    print(root_node)
    if root_node.has_error:
        return False, "解析错误: Julia代码包含语法错误。"

    results = {
        "function_def": [], "loops": [], "conditionals": [], "bindings": [], "others": []
    }
    classify_node(root_node, code_bytes, results)
    return True, results

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

def split_broken_julia_structure(julia_code: str) -> dict:
    def preprocess_julia_code(code: str) -> str:
        """
        把长的一行多语句代码，尽量拆成多行，方便后续分析。
        这里简单用分号拆分，并把关键字前后加换行尝试拆分。
        """
        # 先用分号拆分
        parts = []
        for part in code.split(';'):
            parts.append(part.strip())

        code = '\n'.join(parts)

        # 关键字周围加换行
        keywords = ['function', 'for', 'while', 'if', 'try', 'catch', 'elseif', 'else', 'end']
        for kw in keywords:
            # 前后加换行
            code = code.replace(f' {kw} ', f'\n{kw}\n')
            code = code.replace(f' {kw}(', f'\n{kw}(')
            code = code.replace(f'){kw} ', f'){kw}\n')
            code = code.replace(f'{kw} ', f'{kw}\n')

        # 额外清理多余空行
        lines = [line.strip() for line in code.splitlines()]
        lines = [line for line in lines if line]
        return '\n'.join(lines)
    code = preprocess_julia_code(julia_code)

    function_def = []
    loops = []
    conditionals = []
    bindings = []
    others = []

    lines = code.splitlines()
    stack = []  # (block_type, acc_lines)

    block_keywords = {
        'function': 'function_def',
        'for': 'loops',
        'while': 'loops',
        'if': 'conditionals',
        'try': 'conditionals',
    }

    conditional_sub_keywords = {'elseif', 'else', 'catch', 'finally'}

    def flush_block(block_type, acc_lines):
        code_str = '\n'.join(acc_lines).strip()
        if not code_str:
            return
        if block_type == 'function_def':
            function_def.append(code_str)
        elif block_type == 'loops':
            loops.append(code_str)
        elif block_type == 'conditionals':
            conditionals.append(code_str)

    for line in lines:
        stripped = line.strip()

        # 检查块起始关键字
        found_start = False
        for kw, btype in block_keywords.items():
            if stripped.startswith(kw + ' ') or stripped == kw or stripped.startswith(kw + '('):
                stack.append((btype, [line]))
                found_start = True
                break
        if found_start:
            continue

        # 检查条件子关键字
        if stack and any(stripped.startswith(k) for k in conditional_sub_keywords):
            stack[-1][1].append(line)
            continue

        # 检查end
        if stripped == 'end':
            if stack:
                btype, acc_lines = stack.pop()
                acc_lines.append(line)
                flush_block(btype, acc_lines)
            else:
                others.append(line)
            continue

        # 非关键字语句
        if stack:
            stack[-1][1].append(line)
        else:
            # 顶层赋值判断
            if '=' in stripped and not stripped.startswith('='):
                left = stripped.split('=')[0].strip()
                if left.replace('.', '').replace('_', '').isalnum():
                    bindings.append(line)
                else:
                    others.append(line)
            else:
                others.append(line)

    # 处理残留块
    while stack:
        btype, acc_lines = stack.pop()
        flush_block(btype, acc_lines)

    return {
        "function_def": function_def,
        "loops": loops,
        "conditionals": conditionals,
        "assignments": bindings,
        "others": others,
    }

def print_ast(node, code_bytes, indent=0):
    prefix = "  " * indent
    desc = f"{node.type}"
    if node.is_named:
        desc += f" [{node.start_point} - {node.end_point}]"
        text = code_bytes[node.start_byte:node.end_byte].decode('utf8').strip()
        # 控制最大显示长度
        if len(text) > 60:
            text = text[:60] + "..."
        desc += f": {text}"
    print(prefix + desc)
    for child in node.children:
        print_ast(child, code_bytes, indent + 1)

def split_lua_by_structure(lua_code: str, debug_ast=False):
    # --- 辅助函数 ---
    def node_text(node, code_bytes):
        if not node:
            return ""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    def find_child_by_type(parent_node, target_type, recursive=False):
        if not parent_node:
            return None
        queue = list(parent_node.children)
        while queue:
            node = queue.pop(0)
            if node.type == target_type:
                return node
            if recursive:
                queue.extend(node.children)
        return None

    def print_ast(node, code_bytes, indent=0):
        prefix = "  " * indent
        desc = f"{node.type}"
        if node.is_named:
            desc += f" [{node.start_point} - {node.end_point}]"
            text = node_text(node, code_bytes).strip().replace('\n', ' ')
            if len(text) > 60:
                text = text[:60] + "..."
            desc += f": {text}"
        print(prefix + desc)
        for child in node.children:
            print_ast(child, code_bytes, indent + 1)

    def classify_node(node, code_bytes, results_dict):
        if not node or not node.is_named:
            return

        node_type = node.type

        # 函数定义（包括 local function）
        if node_type in ['function_declaration', 'function_definition', 'function_expression']:
            full_text = node_text(node, code_bytes)
            body_node = find_child_by_type(node, 'block', recursive=True)
            if body_node:
                prefix = code_bytes[node.start_byte:body_node.start_byte].decode('utf8')
                suffix = code_bytes[body_node.end_byte:node.end_byte].decode('utf8')
                results_dict['function_def'].append((prefix + '{}' + suffix).strip())
                classify_node(body_node, code_bytes, results_dict)
            else:
                results_dict['function_def'].append(full_text.strip())
            return

        # 循环
        elif node_type in ['for_statement', 'while_statement', 'repeat_statement']:
            results_dict['loops'].append(node_text(node, code_bytes))
            body_node = find_child_by_type(node, 'block', recursive=True)
            classify_node(body_node, code_bytes, results_dict)
            return

        # 条件语句
        elif node_type == 'if_statement':
            results_dict['conditionals'].append(node_text(node, code_bytes))
            consequence_node = find_child_by_type(node, 'block', recursive=True)
            alternative_node = find_child_by_type(node, 'else', recursive=True)
            classify_node(consequence_node, code_bytes, results_dict)
            classify_node(alternative_node, code_bytes, results_dict)
            return

        # 顶层赋值语句
        elif node_type in ['variable_declaration', 'assignment_statement', 'local_variable_declaration']:
            results_dict['bindings'].append(node_text(node, code_bytes))
            return

        else:
            for child in node.children:
                classify_node(child, code_bytes, results_dict)

            if node.type in ['return_statement', 'call_expression', 'expression_statement']:
                results_dict['others'].append(node_text(node, code_bytes))

    # --- 主逻辑 ---
    code_bytes = lua_code.encode('utf8')
    LUA_LANGUAGE = tree_sitter_lua.language()
    parser = Parser(Language(LUA_LANGUAGE))

    tree = parser.parse(code_bytes)
    root_node = tree.root_node

    if debug_ast:
        print("===== Lua AST Dump =====")
        print_ast(root_node, code_bytes)
        print("========================")

    results = {
        "function_def": [],
        "loops": [],
        "conditionals": [],
        "bindings": [],
        "others": []
    }

    for top_node in root_node.children:
        classify_node(top_node, code_bytes, results)

    return results, True

def split_racket_by_structure(racket_code: str):
    """
    解析Racket代码字符串，并将其扁平化、互不重叠地分类。
    此版本使用了基于精确AST分析的正确分类逻辑。
    """
    lib_path = 'build/racket-parser.so'

    # --- 1. 加载已编译的库 ---
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        language_func = getattr(lib, 'tree_sitter_racket')
        language_func.restype = ctypes.c_void_p
        language_ptr = language_func()
        RACKET_LANGUAGE = Language(language_ptr)
    except (OSError, AttributeError) as e:
        error_msg = (
            f"加载解析器库 '{lib_path}' 失败: {e}\n\n"
            "请确保您已通过命令行成功手动编译了该库。"
        )
        return False, error_msg

    parser = Parser(RACKET_LANGUAGE)

    # --- 辅助函数 ---
    def node_text(node, code_bytes):
        if not node: return ""
        return code_bytes[node.start_byte:node.end_byte].decode('utf8')

    # --- 核心处理函数 (已根据AST修正) ---
    def classify_node(node, code_bytes, results_dict):
        # 忽略注释和非命名节点（如括号）
        if not node or not node.is_named or node.type == 'comment':
            return

        node_type = node.type

        # Case 1: #lang racket 声明
        if node_type == 'extension':
            results_dict['others'].append(node_text(node, code_bytes))
            return

        # Case 2: 所有S-表达式都是一个'list'
        if node_type == 'list':
            # 获取列表中所有有意义的子节点
            children = [child for child in node.children if child.is_named]
            if not children: return

            first_element = children[0]
            # 检查第一个元素是否是决定表达式类型的符号
            if first_element.type != 'symbol':
                results_dict['others'].append(node_text(node, code_bytes))
                return

            op_name = node_text(first_element, code_bytes)

            # --- 开始分类 ---
            if op_name == 'define':
                # (define (func-name ...) body)
                if len(children) > 1 and children[1].type == 'list':
                    sig_node = children[1]
                    results_dict['function_def'].append(f"(define {node_text(sig_node, code_bytes)} ...)")
                    # 递归处理函数体
                    for body_expr in children[2:]:
                        classify_node(body_expr, code_bytes, results_dict)
                # (define var value)
                else:
                    results_dict['bindings'].append(node_text(node, code_bytes))
                return

            elif 'let' in op_name:
                results_dict['bindings'].append(node_text(node, code_bytes))
                # 递归处理let的body部分
                if len(children) > 2:
                    for body_expr in children[2:]:
                        classify_node(body_expr, code_bytes, results_dict)
                return

            elif 'for' in op_name:
                results_dict['loops'].append(node_text(node, code_bytes))
                return

            elif op_name in ['if', 'cond']:
                results_dict['conditionals'].append(node_text(node, code_bytes))
                return

            # 其他所有列表形式（函数调用等）
            else:
                results_dict['others'].append(node_text(node, code_bytes))
                return

        # 如果节点不是list或extension，且未被处理，则归入others
        # （这在顶层不常见，但在函数体内可能出现，如单独的数字或符号）
        elif node_type not in ['program']:
            results_dict['others'].append(node_text(node, code_bytes))

    # --- 主逻辑 ---
    code_bytes = racket_code.encode('utf8')
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    if root_node.has_error:
        return False, "解析错误: Racket代码包含语法错误。"

    results = {
        "function_def": [], "loops": [], "conditionals": [], "bindings": [], "others": []
    }
    # Racket代码的根节点是'program'，其子节点是顶层表达式
    for top_level_expr in root_node.children:
        classify_node(top_level_expr, code_bytes, results)

    return True, results

def inspect_racket_ast(racket_code: str):
    """
    主要功能是解析Racket代码并打印其完整的AST结构用于调试。
    """
    lib_path = 'build/racket-parser.so'

    # --- 1. 加载已编译的库 ---
    try:
        lib = ctypes.cdll.LoadLibrary(lib_path)
        language_func = getattr(lib, 'tree_sitter_racket')
        language_func.restype = ctypes.c_void_p
        language_ptr = language_func()
        RACKET_LANGUAGE = Language(language_ptr)
    except (OSError, AttributeError) as e:
        print(f"加载解析器库 '{lib_path}' 失败: {e}", file=sys.stderr)
        return

    parser = Parser(RACKET_LANGUAGE)
    code_bytes = racket_code.encode('utf8')
    tree = parser.parse(code_bytes)
    root_node = tree.root_node

    # --- 2. 定义AST打印函数 ---
    def node_text_for_print(node, code_bytes):
        # 替换换行符以美化打印
        return code_bytes[node.start_byte:node.end_byte].decode('utf8').replace('\n', '\\n')

    def print_ast(node, code_bytes, indent_level=0):
        indent = "  " * indent_level
        # 打印节点信息：缩进 + 节点类型 + 节点对应的文本
        print(f"{indent}{node.type}: '{node_text_for_print(node, code_bytes)}'")
        for child in node.children:
            print_ast(child, code_bytes, indent_level + 1)

    # --- 3. 打印完整的AST ---
    print("--- Racket Abstract Syntax Tree (AST) ---")
    print_ast(root_node, code_bytes)
    print("\n" + "=" * 40 + "\n")

    if root_node.has_error:
        print("解析错误: Racket代码包含语法错误。", file=sys.stderr)
# --- 测试入口 ---
if __name__ == "__main__":
    test_code_success = """
    #lang racket
    (define (factorial n)
      (if (<= n 1)
          1
          (* n (factorial (- n 1)))))
    (define x 10)
    """
    print("--- 测试Racket代码解析 (自动构建) ---")
    # inspect_racket_ast(test_code_success)


    success_flag, result_data = split_racket_by_structure(test_code_success)

    print(f"\nSuccess Flag: {success_flag}")
    if success_flag:
        print("Result Dictionary:")
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
    else:
        print(f"Result Error:\n{result_data}")

    # result, ok = split_racket_by_structure(test_code_success)
    # import json
    #
    # print(json.dumps(result, indent=2, ensure_ascii=False))