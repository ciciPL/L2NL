import re


def process_code_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        current_index = None
        current_code = []
        in_code_block = False

        for line in f_in:
            line = line.rstrip()  # 去掉右侧空白

            # 检测索引行（如 "1:"）
            if re.match(r'^\d+:', line):
                # 如果已经有收集的代码，先写入
                if current_index is not None and current_code:
                    compressed_code = '\t'.join(current_code)
                    f_out.write(f"{current_index}:\t{compressed_code}\n")

                # 开始新的代码块
                parts = line.split(':', 1)
                current_index = parts[0].strip()
                current_code = []

                # 如果索引行后面直接有代码
                code_part = parts[1].strip()
                if code_part:
                    current_code.append(code_part)

                in_code_block = True
                continue

            # 处理代码块内容
            if in_code_block:
                if line.startswith('```'):
                    # 代码块结束，写入当前收集的代码
                    if current_code:
                        compressed_code = '\t'.join(current_code)
                        f_out.write(f"{current_index}:\t{compressed_code}\n")
                    current_code = []
                    in_code_block = False
                elif line:  # 非空行才收集
                    current_code.append(line.strip())

        # 处理文件末尾可能未处理的代码块
        if current_index and current_code:
            compressed_code = '\t'.join(current_code)
            f_out.write(f"{current_index}:\t{compressed_code}\n")

# 示例调用
process_code_file('../r_result/r_2_python.txt', '../r_result/r_2_python_clean.txt')