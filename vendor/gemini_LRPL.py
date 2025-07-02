import json

import tree_sitter_ocaml
from tree_sitter import Language, Parser
import tree_sitter_julia as julia_lang


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

# --- 测试入口 ---
if __name__ == "__main__":
    test_code_success = """
    function graph_filename_url(site_id, base_graph_name)     fn = joinpath("output", "images", "$(site_id)_$(base_graph_name).png")     url = joinpath("images", "$(site_id)_$(base_graph_name).png")     return fn, url end
"""
    code = f"""
    let do_detect_secrets  : string = "Disabled, hangs when executed from python or gitbash
"""
    print("--- 测试Julia代码解析 ---")

    result_data,success_flag = split_ocaml_by_structure(test_code_success)

    print(f"Success Flag: {success_flag}")
    if success_flag:
        # 为了美观打印，我们使用json.dumps
        print("Result Dictionary:")
        print(json.dumps(result_data, indent=2, ensure_ascii=False))
    else:
        print(f"Result Error: {result_data}")