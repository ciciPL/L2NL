import re

# 提取 <summary>  </summary>中间的值
def extract_summary_context(input_file, output_file):
    with open(input_file, 'r') as f, \
            open(output_file, 'w') as o:
        for line in f:
            index = line.split(':')[0]
            match = re.search(r"<summary>(.*?)</summary>", line)
            if match:
                summary = match.group(1)
                o.write(index + ': ' + summary + '\n')
            else:
                o.write(line)


# #修复数据跨行
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

input_path = '../../experiment/ds-coder-1_3B/r_result/r_2_python_2_NL_sentences_2th.txt'
output_path = '../../experiment/ds-coder-1_3B/r_result/r_2_python_2_NL_sentences_2th_clean.txt'


def fix_multiline_data(data_lines):
    fixed_lines = []
    buffer = ""

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        if line.split(':', 1)[0].strip()=='4820':break
        # 检查是否是新的索引行（格式为"数字:内容"）
        if ':' in line and line.split(':', 1)[0].strip().isdigit():
            if buffer:  # 如果buffer有内容，先保存
                fixed_lines.append(buffer)
                buffer = ""
            buffer = line
        else:
            # 追加到当前buffer（加空格分隔）
            buffer += " " + line if buffer else line

    if buffer:  # 添加最后一条
        fixed_lines.append(buffer)

    return fixed_lines


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()

    fixed_lines = fix_multiline_data(input_lines)
    flag=1
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
        flag+=1

if __name__ == '__main__':
    process_file(input_path, output_path)
    # 示例调用
    # filter_index_lines('../../experiment/ds_1B/r_result/r_NL_sft_without_sentences_2_without_sentencesSFT.txt',
    #                    '../../experiment/ds_1B/r_result/r_NL_sft_without_sentences_2_without_sentencesSFT.txt')
