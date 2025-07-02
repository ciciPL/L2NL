# from tree_sitter import Parser, Language
# import tree_sitter_ocaml
#
# LANG_OCAML = Language(tree_sitter_ocaml.language_ocaml())
# STRUCT_KEYS = ["function_def", "loops", "conditionals", "assignments", "others"]
#
#
# def node_text(source_code, node):
#     return source_code[node.start_byte:node.end_byte].decode("utf8")
#
#
# def split_ocaml_by_structure_debug(code: str):
#     parser = Parser(LANG_OCAML)
#
#     source_code = code.encode("utf8")
#     tree = parser.parse(source_code)
#     root = tree.root_node
#     print("DEBUG root node:",root)
#     results = {k: [] for k in STRUCT_KEYS}
#     used_ranges = []
#
#     def mark_used(node):
#         used_ranges.append((node.start_byte, node.end_byte))
#
#     def is_used(node):
#         return any(start <= node.start_byte < end or start < node.end_byte <= end for start, end in used_ranges)
#
#     def walk(node, depth=0):
#         indent = "  " * depth
#         node_type = node.type
#         text = node_text(source_code, node).strip()
#         print(f"{indent}[Debug] Node type: {node_type}, text: '{text[:60].replace(chr(10), ' ')}...'")
#
#         # 函数定义
#         if node_type == "value_definition":
#             for c in node.children:
#                 if c.type == "let_binding":
#                     print(f"{indent}[Debug] Processing let_binding")
#                     lhs_parts = []
#                     func_body = None
#                     for child in c.children:
#                         ctype = child.type
#                         ctext = node_text(source_code, child).strip()
#                         print(f"{indent}  [Debug] Child type: {ctype}, text: '{ctext[:30]}...'")
#                         if ctype in ["value_name", "parameter", "pattern"]:
#                             lhs_parts.append(ctext)
#                         elif ctype in ["if_expression", "function_expression", "let_expression", "application_expression"]:
#                             func_body = child
#                     if lhs_parts:
#                         func_sig = "let " + " ".join(lhs_parts)
#                         results["function_def"].append(func_sig)
#                         mark_used(node)
#                         if func_body:
#                             print(f"{indent}[Debug] Walking function body node type: {func_body.type}")
#                             walk(func_body, depth + 1)
#                     return
#
#         # 条件语句
#         if node_type == "if_expression":
#             results["conditionals"].append(text)
#             mark_used(node)
#             for c in node.children:
#                 walk(c, depth + 1)
#             return
#
#         # 循环结构（不常见但保留）
#         if node_type in ["for_expression", "while_expression"]:
#             results["loops"].append(text)
#             mark_used(node)
#             for c in node.children:
#                 walk(c, depth + 1)
#             return
#
#         # 赋值表达式
#         if node_type == "let_binding":
#             results["assignments"].append(text)
#             mark_used(node)
#             return
#
#         # 递归向下处理子节点
#         for c in node.children:
#             walk(c, depth + 1)
#
#     walk(root)
#
#     # others: 没被归类的叶子节点
#     def collect_others(node):
#         if node.child_count == 0 and node.is_named and not is_used(node):
#             text = node_text(source_code, node).strip()
#             if text and len(text) > 1 and not any(op in text for op in ['=', '+', '-', '*', '/', '(', ')', '[', ']']):
#                 print(f"[Debug] [others] Add: '{text}'")
#                 results["others"].append(text)
#         for c in node.children:
#             collect_others(c)
#
#     collect_others(root)
#
#     # 去重
#     for k in results:
#         results[k] = sorted(set(results[k]))
#
#     return results, True
#
#
#
#
# if __name__ == "__main__":
#     ocaml_code = """
# let center (bbox : int list) : int list =   let x = (List.nth bbox 0) + (List.nth bbox 2) in   let y = (List.nth bbox 1) + (List.nth bbox 3) in   [x / 2; y / 2] ;;
#
#     """
#     result, success = split_ocaml_by_structure_debug(ocaml_code)
#     print("成功:", success)
#     for key, stmts in result.items():
#         print(f"== {key.upper()} ==")
#         for s in stmts:
#             print("-", s)
