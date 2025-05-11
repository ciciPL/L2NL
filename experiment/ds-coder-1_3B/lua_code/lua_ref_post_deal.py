import re

input_file = "../lua_result/lua_ref.txt"
output_file = "../lua_result/lua_ref_48194.txt"
# 存储结果
results = []

# 读取整个文件
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 分割每个 index: 开头的数据块
pattern_block = r'^(\d+):(.*?)(?=\n\d+:|\Z)'
blocks = re.findall(pattern_block, content, re.DOTALL | re.MULTILINE)

for idx, block in blocks:
    # 找出所有 genSummary: 行
    summaries = re.findall(r'genSummary:\s*(.+?)\s*(?:\n|$)', block, re.DOTALL)

    # 倒序遍历，取第一个不等于 "summary" 的摘要
    valid_summary = None
    for s in reversed(summaries):
        if s.strip().lower() != "summary":
            valid_summary = s.strip()
            break

    if valid_summary:
        results.append(f"{idx}:{valid_summary}")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print(f"成功提取 {len(results)} 条有效摘要到 {output_file}")