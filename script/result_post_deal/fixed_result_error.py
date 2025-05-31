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


# 示例调用
filter_index_lines('PCSD_ref_gen_vllm.txt', 'PCSD_ref_gen_vllm_clean.txt')
