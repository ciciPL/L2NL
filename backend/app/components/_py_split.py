"""Python AST structural splitter.

Verbatim extraction of `split_python_by_structure` from the L2NL research code
(vendor/easc/ast_split_utils.py). Pulled out on its own because the full module
imports tree-sitter language packages we don't need — the pivot is always Python,
and this function only needs stdlib `ast`. Behavior must match the version the
core-block classifier was trained against, so do not "improve" it.
"""
from __future__ import annotations
import ast


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
                except Exception:
                    start = stmt.lineno - 1
                    end = getattr(stmt, 'end_lineno', stmt.lineno)
                    lines = raw_code.splitlines()[start:end]
                    segment = "\n".join(lines)

                if isinstance(stmt, (ast.For, ast.While)):
                    result['loops'].append(segment)
                elif isinstance(stmt, ast.If):
                    result['conditionals'].append(segment)
                elif isinstance(stmt, ast.Assign):
                    result['assignments'].append(segment)
                else:
                    result['others'].append(segment)

        return result, True

    except SyntaxError:
        return {
            'function_def': '',
            'loops': [],
            'conditionals': [],
            'assignments': [],
            'others': []
        }, False
