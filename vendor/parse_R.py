import json
import subprocess
import os

# 这个分隔符必须与 R 脚本中的 STATEMENT_SEPARATOR 完全一致
R_STATEMENT_SEPARATOR = " R_STATEMENT_SEPARATOR_MAGIC_STRING_0123456789 "

def split_r_into_statements(raw_r_code: str, r_script_path: str) -> tuple[list[str], bool]:
    """
    使用 tree-sitter (通过外部R脚本) 将R代码分割成独立的语句。
    R脚本应打印由 R_STATEMENT_SEPARATOR 分隔的语句。

    Args:
        raw_r_code: 要解析的R代码字符串。
        r_script_path: parse_r_statements.R 脚本的路径。

    Returns:
        一个元组，包含：
        - list[str]: 代码语句字符串的列表。
        - bool: 解析是否成功。如果R脚本成功执行并返回了（可能为空的）语句，则为True。
                 如果R脚本执行失败或未找到，则为False，并返回基于换行符的简单分割结果。
    """
    # 如果输入代码本身就是空的或只包含空白，则直接返回空列表和成功
    if not raw_r_code.strip():
        return [], True

    try:
        # 注意：确保 Rscript 在您的 PATH 中，或者提供 Rscript 的完整路径
        # 确保 r_script_path 是正确的
        if not os.path.exists(r_script_path):
            print(f"错误: R 解析脚本 '{r_script_path}' 未找到。")
            # 模仿Python AST解析失败时的回退行为
            fallback_sequences = [line.strip() for line in raw_r_code.strip().split('\n') if line.strip()]
            return fallback_sequences if fallback_sequences else [raw_r_code.strip()], False

        process = subprocess.run(
            ["Rscript", r_script_path, raw_r_code],
            capture_output=True,
            text=True,
            check=True,  # 如果R脚本以非零状态码退出（例如发生错误），则会抛出 CalledProcessError
            encoding="utf-8"
        )
        stdout_from_r = process.stdout.strip() # 移除R脚本最后可能添加的单个换行符

        # 如果R脚本的输出为空字符串，表示没有有效的语句被提取
        # （例如，输入R代码可能只包含注释或空白）
        if not stdout_from_r:
            return [], True # 这种情况视为成功，但没有语句

        # 使用定义好的分隔符来分割R脚本的输出
        statements = stdout_from_r.split(R_STATEMENT_SEPARATOR)

        # 清理每个语句，去除可能存在的前后空白，并过滤掉完全是空字符串的条目
        cleaned_statements = [s.strip() for s in statements if s.strip()]

        return cleaned_statements, True

    except subprocess.CalledProcessError as e:
        print(f"执行R脚本时出错。代码 (前100字符): '{raw_r_code[:100]}...'")
        print(f"Rscript 标准错误输出: {e.stderr}")
        # 解析失败，回退到按行分割 (模仿您Python代码中的行为)
        fallback_sequences = [line.strip() for line in raw_r_code.strip().split('\n') if line.strip()]
        # 如果按行分割结果也为空，但原始代码不为空，则返回原始代码作为一个语句
        return fallback_sequences if fallback_sequences else [raw_r_code.strip()], False
    except FileNotFoundError:
        # Rscript 命令本身未找到
        print("错误: Rscript 命令未找到。请确保R已安装并且Rscript在您的PATH环境变量中。")
        fallback_sequences = [line.strip() for line in raw_r_code.strip().split('\n') if line.strip()]
        return fallback_sequences if fallback_sequences else [raw_r_code.strip()], False
    except Exception as general_error:
        # 捕获其他潜在的意外错误
        print(f"分割R代码时发生未知错误: {general_error}")
        fallback_sequences = [line.strip() for line in raw_r_code.strip().split('\n') if line.strip()]
        return fallback_sequences if fallback_sequences else [raw_r_code.strip()], False


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
        # 打印 stderr 方便调试 R 代码中的 message() 输出
        if process.stderr.strip():
            print("[R stderr]:")
            print(process.stderr.strip())

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


if __name__ == "__main__":
    r_code = """vector_subtract <- function(u, v) {     diff <- c()     for (i in 1:3) {         diff <- c(diff, u[i] - v[i])     }     return(diff) }"""

    r_script_path = "parse_structure.R"
    result, success = split_r_by_structure(r_code, r_script_path)

    print("成功:", success)
    if success:
        for key, stmts in result.items():
            print(f"== {key.upper()} ==")
            for stmt in stmts:
                cleaned_stmt = ' '.join(stmt.split())
                print(cleaned_stmt)
            print("---")