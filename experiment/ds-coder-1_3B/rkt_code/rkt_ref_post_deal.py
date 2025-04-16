input_file = "rkt_ref.txt"
output_file = "rkt_ref_40510.txt"

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        if "genSummary:" in line:
            # 分割索引和内容
            parts = line.split("genSummary:", 1)
            index = parts[0].strip()  # 提取索引（如 "40464:"）
            content = parts[1].strip()  # 提取内容
            # 写入新格式：索引 + \t + 内容
            f_out.write(f"{index}\t{content}\n")
        else:
            # 处理无 genSummary: 的行（直接保留）
            f_out.write(line)