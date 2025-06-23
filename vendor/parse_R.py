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

# --- 如何在您的 make_pcsd_dataset_modified 函数中使用 ---
#
# 假设您将上述 R 脚本保存为 "parse_r_statements.R"
# 并且 `split_r_into_statements` 函数也已定义。
#
# 您需要修改 `make_pcsd_dataset_modified` 函数（或创建一个新版本，如 `make_pcsd_dataset_for_r`）：
#
# 1. 将调用 `split_python_with_ast(raw_code)` 的地方替换为：
#    `code_statement_strings, ast_success = split_r_into_statements(raw_code, "path/to/your/parse_r_statements.R")`
#
# 2. `split_identifier_into_parts(stmt_str_cleaned)` 这部分是针对Python标识符的。
#    对于R语言，您可能需要一个不同的逻辑来分割标识符（R的标识符可以包含点 `.`）。
#    或者，您可以暂时简化处理，例如，直接将整个R语句转为小写，或者使用更通用的基于非字母数字字符的分割。
#    例如，一个非常简单的替代品可能是：
#    `stmt_lower = stmt_str_cleaned.lower()` (不进一步分割，只转小写)
#    或者
#    `r_tokens = re.split(r'[^a-zA-Z0-9_.]+', stmt_str_cleaned)` (一个粗略的分割)
#    `stmt_lower = ' '.join(r_tokens).lower()`
#
# 下面是一个演示如何使用的示例：
if __name__ == "__main__":
    # 将 parse_r_statements.R 脚本放在与此Python脚本相同的目录下，或提供正确路径
    r_parser_script = "parse_fixed.R" # 或者 "path/to/your/parse_r_statements.R"

    sample_r_code_1 = """
    dela0 <- function(n, a0, a1, x, y) {   sum(a0 + a1 * x - y) / n }
    """

    print(f"--- 解析 R 代码 1 ---")
    statements1, success1 = split_r_into_statements(sample_r_code_1, r_parser_script)
    print(f"成功: {success1}")
    if success1:
        for i, stmt in enumerate(statements1):
            print(f"语句 {i+1}: {stmt}")
    else:
        print(f"回退的语句: {statements1}")

