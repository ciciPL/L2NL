import os
import json


def process_pcsd_dataset(base_dir, output_dir):
    """
    处理 PCSD 数据集，转换为 DeepSeek Coder 训练所需的 JSONL 格式

    参数:
    - base_dir: 包含 train/valid/test 目录的基础路径
    - output_dir: 输出 JSONL 文件的目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 数据集划分
    splits = ['train', 'valid', 'test']

    for split in splits:
        # 构建输入文件路径
        code_file = os.path.join(base_dir, split, 'code')
        nl_file = os.path.join(base_dir, split, 'nl')

        # 构建输出 JSONL 文件路径
        output_file = os.path.join(output_dir, f'{split}_pcsd.jsonl')

        # 处理数据
        with open(code_file, 'r', encoding='utf-8') as f_code, \
                open(nl_file, 'r', encoding='utf-8') as f_nl, \
                open(output_file, 'w', encoding='utf-8') as f_out:

            for code, nl in zip(f_code, f_nl):
                code = code.strip()
                nl = nl.strip()

                # 构建 DeepSeek Coder 训练所需的 JSON 格式
                entry = {
                    "instruction": "Provide a concise summary of the following code:",
                    "output": nl
                }

                # 写入 JSONL 文件
                f_out.write(json.dumps(entry) + '\n')

                # 打印处理信息
        print(f"Processed {split} split. Output file: {output_file}")

    # 使用示例


base_dir = '../dataset/Clean_PCSD'  # 根据实际路径调整
output_dir = '../dataset/PCSD'
process_pcsd_dataset(base_dir, output_dir)