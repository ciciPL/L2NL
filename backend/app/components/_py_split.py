"""Python AST structural splitter.

Adapted from the recursive `split_python_by_structure_new` path in the L2NL
research code (vendor/easc/ast_split_utils.py). The plugin receives Python
pivots that may be async functions, class methods, or nested helpers, so the
splitter must drill into function/class bodies instead of treating the whole
container as one statement.
"""
from __future__ import annotations
import ast


def split_python_by_structure(raw_code):
    try:
        tree = ast.parse(raw_code)
        result = {
            'function_def': [],
            'loops': [],
            'conditionals': [],
            'assignments': [],
            'others': []
        }

        def _segment(stmt):
            try:
                segment = ast.get_source_segment(raw_code, stmt)
            except Exception:
                start = stmt.lineno - 1
                end = getattr(stmt, 'end_lineno', stmt.lineno)
                lines = raw_code.splitlines()[start:end]
                segment = "\n".join(lines)
            return segment or ""

        def _extract_body(body):
            for stmt in body:
                segment = _segment(stmt)
                if not segment:
                    continue

                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result['function_def'].append(segment.split('\n')[0])
                    _extract_body(stmt.body)
                elif isinstance(stmt, ast.ClassDef):
                    _extract_body(stmt.body)
                elif isinstance(stmt, (ast.For, ast.While)):
                    result['loops'].append(segment)
                elif isinstance(stmt, ast.If):
                    result['conditionals'].append(segment)
                elif isinstance(stmt, ast.Assign):
                    result['assignments'].append(segment)
                else:
                    result['others'].append(segment)

        _extract_body(tree.body)

        return result, True

    except SyntaxError:
        return {
            'function_def': [],
            'loops': [],
            'conditionals': [],
            'assignments': [],
            'others': []
        }, False
