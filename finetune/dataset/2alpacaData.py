import json

# 输入和输出文件路径
input_file_path = 'PCSD/train_pcsd.jsonl'  # 替换为你的输入文件路径
output_file_path = 'alpacaPCSD/train_pcsd.json'  # 替换为你想要的输出文件路径

# 系统提示词（适合摘要任务）
system_prompt = "Please generate a concise summary based on the user's instruction."

# 创建一个列表来存储转换后的数据
alpaca_data = []

# 打开输入文件并读取内容
with open(input_file_path, 'r', encoding='utf-8') as infile:
    for line in infile:
        # 解析 JSONL 行
        data = json.loads(line)

        # 提取 instruction 和 output
        instruction = data.get('instruction', '')
        output = data.get('output', '')

        # 创建 Alpaca 格式的字典
        alpaca_format = {
            "instruction": instruction,
            "input": "",  # 用户输入（选填），这里留空
            "output": output,
            "system": system_prompt,
            "history": []  # 历史记录留空
        }

        # 将 Alpaca 格式的字典添加到列表中
        alpaca_data.append(alpaca_format)

    # 将列表写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(alpaca_data, outfile, ensure_ascii=False, indent=2)

print(f"Conversion completed. Output file saved at: {output_file_path}")