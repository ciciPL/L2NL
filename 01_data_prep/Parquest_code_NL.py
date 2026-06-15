"""
Extract code + comment from a MultiPL-T parquet file (R-formatted blocks).

For each `content` cell in the parquet, split lines into:
  - comment lines starting with `#`  -> folded into a single space-joined string
  - everything else (code)           -> folded into a single space-joined string

Outputs two index-prefixed txt files:
  doc_<lang>.txt    -> "<row_index>:<comment text>"
  code_<lang>.txt   -> "<row_index>:<code text>"
"""

import pandas as pd


# ========= R parser =========
def parse_r_block_to_space_joined_strings(content_block_str):
    """
    解析单个 R content 字符串块。
    - 提取 '#' 开头的注释文本部分。
    - 保留原始代码行。
    - 将注释行与代码行分别折叠为空格连接的单行。
    """
    if not isinstance(content_block_str, str):
        return None, None

    collected_comment_text_parts = []
    collected_original_code_lines = []

    for line in content_block_str.split('\n'):
        stripped_line = line.lstrip()
        if stripped_line.startswith('#'):
            comment_text = stripped_line[1:].lstrip()
            collected_comment_text_parts.append(comment_text)
        else:
            collected_original_code_lines.append(line)

    folded_comment = " ".join(collected_comment_text_parts) if collected_comment_text_parts else None
    folded_code = " ".join(collected_original_code_lines) if collected_original_code_lines else None
    return folded_comment, folded_code
# 主要的提取和文件保存函数
def extract_and_save_indexed_folded_content(input_parquet_path, comment_output_path, code_output_path):
    """
    从Parquet文件读取 R 内容，处理每个 content 块，分离注释和代码。
    将注释和代码分别折叠为单行空格连接的字符串。
    即使没有注释，也写入空字符串，保证索引严格对应。
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"错误: 读取 Parquet 文件 '{input_parquet_path}' 失败: {e}")
        return

    if 'content' not in df.columns:
        print(f"错误: 在 Parquet 文件中未找到 'content' 列。")
        return

    total_parquet_entries = len(df)
    print(f"Parquet 文件总条数: {total_parquet_entries}")

    all_final_folded_comment_entries = []
    all_final_folded_code_entries = []

    # 处理每个 block
    for content_item_str in df['content']:
        block_comment_str, block_code_str = parse_r_block_to_space_joined_strings(content_item_str)

        # 即使没有注释，也写入空字符串
        all_final_folded_comment_entries.append(block_comment_str if block_comment_str is not None else "")
        all_final_folded_code_entries.append(block_code_str if block_code_str is not None else "")

    # 写入注释文件
    try:
        with open(comment_output_path, 'w', encoding='utf-8') as f_comment:
            for i, entry_str in enumerate(all_final_folded_comment_entries, 1):
                entry_str = entry_str.replace('\n', ' ').replace('\r', ' ').strip()
                f_comment.write(f"{i}:{entry_str}\n")
    except IOError as e:
        print(f"错误: 写入注释文件 '{comment_output_path}' 失败: {e}")

    # 写入代码文件
    try:
        with open(code_output_path, 'w', encoding='utf-8') as f_code:
            for i, entry_str in enumerate(all_final_folded_code_entries, 1):
                entry_str = entry_str.replace('\n', ' ').replace('\r', ' ').strip()
                f_code.write(f"{i}:{entry_str}\n")
    except IOError as e:
        print(f"错误: 写入代码文件 '{code_output_path}' 失败: {e}")

    print(f"处理完成。")
    print(f"折叠后注释条目: {len(all_final_folded_comment_entries)}")
    print(f"折叠后代码条目: {len(all_final_folded_code_entries)}")



# 主执行块
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract index-prefixed comment / code from a MultiPL-T parquet file."
    )
    parser.add_argument("--input", default="./data/LowData/r/r-00000-of-00001.parquet",
                        help="Input parquet file path")
    parser.add_argument("--output_comment", default="./data/LowData/r/doc_r.txt",
                        help="Output comment txt path")
    parser.add_argument("--output_code", default="./data/LowData/r/code_r.txt",
                        help="Output code txt path")
    args = parser.parse_args()

    input_parquet_file = args.input
    output_comment_file = args.output_comment
    output_code_file = args.output_code

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