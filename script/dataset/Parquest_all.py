import pandas as pd


def convert_parquet_content_to_indexed_single_lines(input_parquet_path, output_text_file_path):
    """
    读取 Parquet 文件中的 'content' 列。每个 'content' 条目（可能包含多行）
    通过将其内部的换行符替换为单个空格来转换为单行文本。
    结果以 "index:单行内容" 的格式保存到指定的文本文件中。

    Args:
        input_parquet_path (str): 输入的 Parquet 文件路径。
        output_text_file_path (str): 输出的文本文件路径。
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"错误: 读取 Parquet 文件 '{input_parquet_path}' 失败: {e}")
        return

    if 'content' not in df.columns:
        print(f"错误: 在 Parquet 文件中未找到 'content' 列。")
        return

    all_output_entries = []  # 用于存储最终要写入文件的 "index:内容" 字符串

    # 遍历 'content' 列中的每一个条目
    current_index = 1
    for content_entry_str in df['content']:
        single_line_version_of_content = ""  # 初始化为空字符串

        if not isinstance(content_entry_str, str):
            # 对于非字符串类型的内容，我们将其视为空内容进行处理
            # 这样可以确保原始Parquet中的每一行都在输出文件中有一个对应的（可能是空的）条目
            pass  # single_line_version_of_content 保持 ""
        else:
            # 将原始的多行字符串按换行符分割成行列表
            lines = content_entry_str.split('\n')
            # 用单个空格连接这些行，形成单行版本的字符串
            single_line_version_of_content = " ".join(lines)

            # 可选：进行空格规范化处理
            # 如果您希望将多个连续空格（可能由空行或原始文本中的多余空格产生）
            # 替换为单个空格，并去除结果字符串首尾的空格，可以取消下面这行的注释。
            # single_line_version_of_content = re.sub(r'\s+', ' ', single_line_version_of_content).strip()

            # 另一个简单的清理是只去除首尾空格：
            # single_line_version_of_content = single_line_version_of_content.strip()

        # 格式化为 "index:内容"
        all_output_entries.append(f"{current_index}:{single_line_version_of_content}")
        current_index += 1

    # 将所有处理后的带索引的行写入输出文件
    try:
        with open(output_text_file_path, 'w', encoding='utf-8') as f_out:
            for indexed_line in all_output_entries:
                f_out.write(indexed_line + '\n')

        print(f"处理完成。数据已保存到: {output_text_file_path}")
        print(f"总共处理并输出了 {len(all_output_entries)} 个条目。")
    except IOError as e:
        print(f"错误: 写入输出文件 '{output_text_file_path}' 失败: {e}")


# 主执行块
if __name__ == '__main__':
    # 用户需要配置这些路径
    input_parquet_file = 'lua/lua-00000-of-00001.parquet'  # <--- 请修改为您的输入文件路径

    # 定义输出文件名，以反映其内容和格式
    output_single_line_file = 'lua/lua_all.txt'

    convert_parquet_content_to_indexed_single_lines(input_parquet_file, output_single_line_file)