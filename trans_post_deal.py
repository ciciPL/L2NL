genpython_blocks = {}
current_block = []
in_genpython = False
file_path = 'dataset/rkt_result/rkt_python_40510_DS_1-3b.txt'
out_file = 'dataset/rkt_result/rkt_python_clean_DS_1-3b.txt'
current_id = None
indent_level = 0

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # 检测条目ID（数字开头后跟冒号）
        if ':' in line:
            possible_id = line.split(':', 1)[0].strip()
            if possible_id.isdigit():
                if current_id and in_genpython and current_block:
                    code = ''.join(current_block).strip()
                    genpython_blocks[current_id] = code
                current_id = possible_id
                in_genpython = False
                current_block = []
                indent_level = 0

        # 检测genPython开始
        if line.startswith('genPython:') and not in_genpython and current_id:
            in_genpython = True
            current_block = []
            indent_level = 0
        elif in_genpython:
            # 计算当前缩进级别（仅对非空行）
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_block:  # 不是第一行
                    if current_indent > indent_level:
                        indent_level = current_indent

            # 终止条件：遇到相同或更少缩进的新条目开始
            if (':' in line and line.split(':', 1)[0].strip().isdigit() and
                    (len(line) - len(line.lstrip())) <= indent_level):
                in_genpython = False
                if current_block:
                    code = ''.join(current_block).strip()
                    genpython_blocks[current_id] = code
                current_block = []
            else:
                current_block.append(line)

    # 处理最后一个代码块
    if current_id and in_genpython and current_block:
        code = ''.join(current_block).strip()
        genpython_blocks[current_id] = code

# 检查完整性
expected_ids = set(str(i) for i in range(1, 10883))
missing_ids = expected_ids - set(genpython_blocks.keys())

# 写入文件
with open(out_file, 'w', encoding='utf-8') as f:
    # 按数字顺序排序
    for entry_id in sorted(genpython_blocks.keys(), key=int):
        code = genpython_blocks[entry_id]
        code = code.split('note:')[0]
        code = code.split('Note:')[0]
        # 保留代码结构但转换为单行表示
        escaped_code = code.replace('\n', '\\n').replace('    ', '\\t')
        f.write(f"{entry_id}: {escaped_code}\n")

# 输出统计信息
print(f"原始条目数: 10882")
print(f"成功提取数: {len(genpython_blocks)}")
print(f"缺失条目数: {len(missing_ids)}")
if missing_ids:
    with open('dataset/rkt_result/miss_id', 'w', encoding='utf-8') as f:
        for missing_id in missing_ids:
            f.write(str(missing_id)+'\n')