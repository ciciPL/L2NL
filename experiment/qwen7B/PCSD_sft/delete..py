import re


def normalize_sentence(sentence):
    # 只去掉句号前面的空格（可以有多个空格）
    sentence = re.sub(r'\s+\.', '.', sentence)
    # 如果结尾没有句号，则加上一个
    if sentence and not sentence.rstrip().endswith('.'):
        sentence = sentence.rstrip() + '.'
    return sentence


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line in infile:
            normalized = normalize_sentence(line)
            outfile.write(normalized + '\n')

def remove_empty_duplicate_lines(input_file_path, output_file_path):
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        non_empty_lines = [line for line in lines if line.strip()]
        unique_lines = []
        for line in non_empty_lines:
            if line not in unique_lines:
                unique_lines.append(line)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(unique_lines)
        print(f"已成功删除空行和重复行，结果保存于 {output_file_path}")
    except FileNotFoundError:
        print("错误：未找到输入文件！")
    except Exception as e:
        print(f"错误：出现未知错误：{e}")

# 示例调用：
input_file = 'rkt_python_nl_7B_sft_4051.txt'     # 替换为你的输入文件路径
output_file = 'rkt_python_nl_7B_sft_4051_delete.txt'   # 替换为你想保存的输出文件路径

process_file(input_file, output_file)
remove_empty_duplicate_lines(output_file, output_file)
print(f"处理完成，结果已保存到 {output_file}")