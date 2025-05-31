import os

# --- （这是我们之前定义的函数，稍作调整或直接使用其输出来判断） ---
def analyze_special_char_ratio(
    text: str,
    custom_special_chars: str = "（）&……%￥#@！{}[]<>?？，。；：‘’“”\"'~`!$^*-+=|\\/"
) -> tuple[float, int, int]:
    """
    分析字符串中特殊字符的占比。

    参数:
    text (str): 需要分析的输入字符串。
    custom_special_chars (str): 一个包含所有被视作特殊字符的字符串。

    返回:
    tuple[float, int, int]:
        - 特殊字符的实际占比 (百分比)。
        - 特殊字符的数量。
        - 字符串总长度。
    """
    if not text: # 处理空字符串的情况
        return 0.0, 0, 0

    special_char_set = set(custom_special_chars)
    special_char_count = 0
    for char in text:
        if char in special_char_set:
            special_char_count += 1

    total_length = len(text)
    if total_length == 0:
        percentage = 0.0
    else:
        percentage = (special_char_count / total_length) * 100

    return percentage, special_char_count, total_length
# --- （函数结束） ---

def process_file_based_on_content_quality(
    input_filepath: str,
    output_filepath: str,
    discard_threshold_percent: float = 10.0,
    special_chars_definition: str = "（）&……%￥#@！{}[]<>?？，。；：‘’“”\"'~`!$^*-+=|\\/"
):
    """
    读取输入文件，检查每行内容的特殊字符占比，
    如果超出阈值则丢弃，否则写入输出文件。

    参数:
    input_filepath (str): 输入文件的路径。
    output_filepath (str): 输出文件的路径。
    discard_threshold_percent (float): 特殊字符占比的丢弃阈值（百分比）。
                                      如果占比 > threshold, 则丢弃。
    special_chars_definition (str): 定义哪些是特殊字符。
    """
    lines_processed = 0
    lines_written = 0
    lines_discarded = 0

    print(f"开始处理文件: '{input_filepath}'")
    print(f"输出到文件: '{output_filepath}'")
    print(f"特殊字符丢弃阈值: > {discard_threshold_percent}%")
    print(f"定义的特殊字符集: '{special_chars_definition}'\n")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as infile, \
             open(output_filepath, 'w', encoding='utf-8') as outfile:

            for line_number, raw_line in enumerate(infile, 1):
                lines_processed += 1
                line = raw_line.strip() # 去除行首尾的空白

                if not line: # 跳过空行
                    # outfile.write(raw_line) # 如果希望保留空行，取消注释此行
                    # print(f"第 {line_number} 行: 空行，已跳过 (或按需保留)。")
                    continue

                parts = line.split(':', 1) # 只在第一个冒号处分割
                if len(parts) == 2:
                    index_part, content_part = parts
                    content_part = content_part.strip() # 内容部分也可能需要去除首尾空白

                    if not content_part: # 如果内容部分为空
                        # print(f"第 {line_number} 行: 内容为空，索引为 '{index_part}'。正在写入...")
                        outfile.write(raw_line) # 保留这类行
                        lines_written += 1
                        continue

                    percentage, spec_count, total_len = analyze_special_char_ratio(
                        content_part,
                        custom_special_chars=special_chars_definition
                    )

                    if percentage > discard_threshold_percent:
                        lines_discarded += 1
                        print(f"第 {line_number} 行: 丢弃。索引: '{index_part}', "
                              f"内容: '{content_part[:50]}...' (特殊字符占比: {percentage:.2f}%, "
                              f"{spec_count}/{total_len})")
                    else:
                        outfile.write(raw_line) # 写入原始行（包含换行符）
                        lines_written += 1
                        # print(f"第 {line_number} 行: 保留。索引: '{index_part}', 特殊字符占比: {percentage:.2f}%")
                else:
                    # 行不符合 "index:内容" 格式
                    lines_discarded += 1 # 或者你可以选择写入，或记录到错误文件
                    print(f"第 {line_number} 行: 格式不符 '{line[:50]}...'，已丢弃。")

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_filepath}' 未找到。")
        return
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return

    print("\n--- 处理完成 ---")
    print(f"总处理行数: {lines_processed}")
    print(f"写入行数: {lines_written}")
    print(f"丢弃行数: {lines_discarded}")

# --- 示例用法 ---
if __name__ == "__main__":
    # 1. 创建一个示例输入文件
    sample_input_filename = "fixedRef.txt"
    sample_output_filename = "fixedRef_clean.txt"

    # 定义你自己的特殊字符，如果需要的话
    my_special_chars = "（）&……%￥#@！{}[]<>?？，。；：‘’“”\"'~`!$^*-+=|\\/." # 添加了英文句号


    # 2. 调用处理函数
    process_file_based_on_content_quality(
        input_filepath=sample_input_filename,
        output_filepath=sample_output_filename,
        discard_threshold_percent=5.0, # 例如，如果特殊字符超过25%则丢弃
        special_chars_definition=my_special_chars
    )
