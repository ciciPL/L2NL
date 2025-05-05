import pandas as pd
import json

# 读取Excel文件，明确指定第一行为标题行
df = pd.read_excel("aliPCSDData/test_500.xlsx", header=0)  # 第一行是表头

# 准备输出数据列表
output_data = []

# 遍历每一行数据（此时索引从0开始对应的是第二行实际数据）
for idx, row in df.iterrows():
    # 获取prompt内容（假设列名为"prompt"）
    prompt_content = row["Prompt"]

    # 构建JSON对象
    json_obj = {
        "custom_id": f"request-{idx + 1}",  # idx从0开始，+1后变为1~500
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "qwen-max",
            "messages": [
                {"role": "system", "content": "你是计算小助手."},
                {"role": "user", "content": prompt_content}
            ]
        }
    }

    # 添加到输出列表
    output_data.append(json_obj)

# 写入JSONL文件
with open("output.jsonl", "w", encoding="utf-8") as f:
    for item in output_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("转换完成！已生成output.jsonl文件")