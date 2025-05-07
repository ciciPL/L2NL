import json

# 输入文件路径
input_file = '../PCSD/PCSD_deep.json'

# 输出文件路径
instruction_output_file = 'alicode.txt'
output_output_file = 'aliref.txt'

# 读取 JSON 数据
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 instruction 到 instruction.txt
with open(instruction_output_file, 'w', encoding='utf-8') as f:
    for idx, item in enumerate(data):
        f.write(f"{idx+1}: {item['instruction']}\n")

# 写入 output 到 output.txt
with open(output_output_file, 'w', encoding='utf-8') as f:
    for idx, item in enumerate(data):
        f.write(f"{idx+1}: {item['output']}\n")

print("数据已成功写入 instruction.txt 和 output.txt")