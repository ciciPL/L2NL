import pandas as pd
import re  # 导入 re 模块，以备后续可能需要的更复杂的空格处理


# 辅助函数，用于处理从Parquet读取的单个 content 字符串块
def parse_lua_block_to_space_joined_strings(content_block_str):
    """
    解析单个Lua content字符串块。
    - 提取注释文本部分（'--'之后的内容）。
    - 保留原始代码行。
    - 将此块内的所有注释文本部分用单个空格连接成一个字符串。
    - 将此块内的所有原始代码行用单个空格连接成一个字符串。
    返回一个元组: (块内所有注释连接成的字符串, 块内所有代码连接成的字符串)。
    如果 content_block_str 不是字符串，则返回 (None, None)。
    如果块内没有相应的注释或代码，则对应的返回字符串为 None。
    """
    if not isinstance(content_block_str, str):
        return None, None  # 如果内容不是字符串，则跳过

    collected_comment_text_parts = []  # 用于存储此块中每条注释行的文本内容
    collected_original_code_lines = []  # 用于存储此块中每条原始代码行

    original_lines_in_block = content_block_str.split('\n')

    for line in original_lines_in_block:
        stripped_line_for_prefix_check = line.lstrip()  # 去除前导空格以检查 '--'

        if stripped_line_for_prefix_check.startswith('--'):
            # 此行为注释
            comment_text = stripped_line_for_prefix_check[2:]  # 获取 '--' 之后的内容
            if comment_text.startswith(' '):  # 如果 '-- ' 这样有一个空格，也去掉
                comment_text = comment_text[1:]
            collected_comment_text_parts.append(comment_text)
        else:
            # 此行为代码行或非注释的空行
            collected_original_code_lines.append(line)  # 保留原始行，包括其缩进

    # 创建此块的单行折叠注释字符串
    folded_comment_string_for_block = None
    if collected_comment_text_parts:  # 仅当此块实际包含注释行时才创建字符串
        folded_comment_string_for_block = " ".join(collected_comment_text_parts)
        # 可选的空格规范化 (如果需要将多个空格合并为单个，并去除首尾空格):
        # folded_comment_string_for_block = re.sub(r'\s+', ' ', folded_comment_string_for_block).strip()

    # 创建此块的单行折叠代码字符串
    folded_code_string_for_block = None
    if collected_original_code_lines:  # 仅当此块实际包含代码行时才创建字符串
        folded_code_string_for_block = " ".join(collected_original_code_lines)
        # 可选的空格规范化:
        # folded_code_string_for_block = re.sub(r'\s+', ' ', folded_code_string_for_block).strip()

    return folded_comment_string_for_block, folded_code_string_for_block


# 主要的提取和文件保存函数
def extract_and_save_indexed_folded_content(input_parquet_path, comment_output_path, code_output_path):
    """
    从Parquet文件读取Lua内容，处理每个content块以分离注释和代码，
    将每个块的注释部分和代码部分分别折叠成单行（通过空格连接），
    并为每行添加 "index:" 前缀后，保存到各自的输出文件中。
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"错误: 读取 Parquet 文件 '{input_parquet_path}' 失败: {e}")
        return

    if 'content' not in df.columns:
        print(f"错误: 在 Parquet 文件中未找到 'content' 列。")
        return

    all_final_folded_comment_entries = []
    all_final_folded_code_entries = []

    # 处理Parquet 'content'列中的每个条目
    for content_item_str in df['content']:
        block_comment_str, block_code_str = parse_lua_block_to_space_joined_strings(content_item_str)

        if block_comment_str is not None:
            all_final_folded_comment_entries.append(block_comment_str)

        if block_code_str is not None:
            all_final_folded_code_entries.append(block_code_str)

    # 将带索引的折叠后注释写入输出文件
    try:
        with open(comment_output_path, 'w', encoding='utf-8') as f_comment:
            for i, entry_str in enumerate(all_final_folded_comment_entries, 1):  # 索引从1开始
                f_comment.write(f"{i}:{entry_str}\n")
    except IOError as e:
        print(f"错误: 写入注释文件 '{comment_output_path}' 失败: {e}")

    # 将带索引的折叠后代码写入输出文件
    try:
        with open(code_output_path, 'w', encoding='utf-8') as f_code:
            for i, entry_str in enumerate(all_final_folded_code_entries, 1):  # 索引从1开始
                f_code.write(f"{i}:{entry_str}\n")
    except IOError as e:
        print(f"错误: 写入代码文件 '{code_output_path}' 失败: {e}")


# 主执行块
if __name__ == '__main__':
    # 用户需要配置这些路径
    input_parquet_file = 'lua/lua-00000-of-00001.parquet'  # <--- 请修改为您的输入文件路径

    # 更新输出文件名以反映包含索引
    output_comment_file = 'lua/comment.txt'
    output_code_file = 'lua/code.txt'

    extract_and_save_indexed_folded_content(input_parquet_file, output_comment_file, output_code_file)

    print(f"处理完成。")

    # (可选) 打印生成文件的条目计数
    try:
        with open(output_comment_file, 'r', encoding='utf-8') as f_comment_count:
            num_comment_entries = sum(1 for _ in f_comment_count)
        print(f"带索引的折叠后注释条目已保存到: {output_comment_file} (共 {num_comment_entries} 个条目)")
    except FileNotFoundError:
        print(f"提示: 注释文件 '{output_comment_file}' 可能未生成或为空。")

    try:
        with open(output_code_file, 'r', encoding='utf-8') as f_code_count:
            num_code_entries = sum(1 for _ in f_code_count)
        print(f"带索引的折叠后代码条目已保存到: {output_code_file} (共 {num_code_entries} 个条目)")
    except FileNotFoundError:
        print(f"提示: 代码文件 '{output_code_file}' 可能未生成或为空。")