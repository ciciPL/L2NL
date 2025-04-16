import re


def filter_index_lines(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        # 正则匹配以数字+冒号开头的行（如 "40498:"）
        index_pattern = re.compile(r'^\d+:')

        for line in f_in:
            line = line.strip()
            # 检查是否符合索引行格式
            if index_pattern.match(line):
                # 去除冗余部分（如 "(code)" -> "..."）
                clean_line = line.split('(code)')[0].strip()
                f_out.write(f"{clean_line}\n")


# 示例调用
filter_index_lines('rkt_2_python_2_NL.txt', 'rkt_2_python_2_NL_40510.txt')