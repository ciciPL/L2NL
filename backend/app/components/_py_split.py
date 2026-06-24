"""Python AST structural splitter.

The production plugin follows the research-code entry point
`split_python_by_structure_new` from vendor/easc/ast_split_utils.py. That path
supports multiple/nested Python functions by collecting every function signature
and recursing into function bodies instead of treating the whole function as one
statement.
"""
from __future__ import annotations
import ast
import io
import tokenize


def _slice_segment(lines, start, end):
    start_line, start_col = start
    end_line, end_col = end
    if start_line == end_line:
        return lines[start_line - 1][start_col:end_col]
    selected = lines[start_line - 1:end_line]
    selected[0] = selected[0][start_col:]
    selected[-1] = selected[-1][:end_col]
    return "\n".join(selected)


def _function_signature(segment: str) -> str:
    lines = segment.splitlines()
    started = False
    start = (1, 0)
    pending_async_start = None
    depth = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(segment).readline):
            text = tok.string
            if not started:
                if tok.type == tokenize.NAME and text == "async":
                    pending_async_start = tok.start
                    continue
                if tok.type == tokenize.NAME and text == "def":
                    start = pending_async_start or tok.start
                    started = True
                else:
                    pending_async_start = None
                    continue

            if text in "([{":
                depth += 1
            elif text in ")]}" and depth:
                depth -= 1
            elif text == ":" and depth == 0:
                return _slice_segment(lines, start, tok.end).strip()
    except tokenize.TokenError:
        pass
    return segment.split("\n")[0].strip()


def split_python_by_structure_new(raw_code):
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
                    result['function_def'].append(_function_signature(segment))
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


def split_python_by_structure(raw_code):
    """Backward-compatible alias for older plugin code paths."""
    return split_python_by_structure_new(raw_code)
