# import os
# import json
#
#
# def process_pcsd_dataset(base_dir, output_dir):
#     """
#     处理 PCSD 数据集，转换为 DeepSeek Coder 训练所需的 JSONL 格式
#
#     参数:
#     - base_dir: 包含 train/valid/test 目录的基础路径
#     - output_dir: 输出 JSONL 文件的目录
#     """
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 数据集划分
#     splits = ['train', 'valid', 'test']
#
#     for split in splits:
#         # 构建输入文件路径
#         code_file = os.path.join(base_dir, split, 'code')
#         nl_file = os.path.join(base_dir, split, 'nl.txt')
#
#         # 构建输出 JSONL 文件路径
#         output_file = os.path.join(output_dir, f'{split}_pcsd.jsonl')
#
#         # 处理数据
#         with open(code_file, 'r', encoding='utf-8') as f_code, \
#                 open(nl_file, 'r', encoding='utf-8') as f_nl, \
#                 open(output_file, 'w', encoding='utf-8') as f_out:
#
#             for code, nl in zip(f_code, f_nl):
#                 code = code.strip()
#                 nl = nl.strip()
#
#                 # # 构建 DeepSeek Coder 训练所需的 JSON 格式
#                 # entry = {
#                 #     "instruction": "Provide a concise summary of the following code:"+code,
#                 #     "output": nl.txt
#                 # }
#
#                 # 构建符合阿里云练的训练集消息列表
#                 messages = [
#                     {"role": "system", "content": "You are a helpful assistant for code summary."},
#                     {"role": "user", "content": code},
#                     {"role": "assistant", "content": nl}
#                 ]
#                 entry = {"messages": messages}
#
#                 # 写入 JSONL 文件
#                 f_out.write(json.dumps(entry) + '\n')
#
#                 # 打印处理信息
#         print(f"Processed {split} split. Output file: {output_file}")
#
#     # 使用示例
#
#
# base_dir = 'Clean_PCSD'  # 根据实际路径调整
# output_dir = 'aliPCSDData'
# process_pcsd_dataset(base_dir, output_dir)

#构建阿里的测试集数据格式
import re
from openpyxl.workbook import Workbook
# 构建阿里的测试集数据格式
import re

from openpyxl.workbook import Workbook


def clean_string(s):
    # 移除非法字符（Excel 不支持的字符）
    # 只保留常见的可打印字符和换行符
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)
    return s.strip()

def generate_prompt_completion_excel(code_file_path, nl_file_path, output_excel_path, max_rows=None):
    # 创建一个新的工作簿和激活的工作表
    wb = Workbook()
    ws = wb.active
    ws.title = "Prompt-Completion"

    # 写入表头
    ws.append(["Prompt", "Completion"])

    # 读取文件并写入 Excel
    with open(code_file_path, 'r', encoding='utf-8', errors='ignore') as code_file, \
         open(nl_file_path, 'r', encoding='utf-8', errors='ignore') as nl_file:

        for i, (code_line, nl_line) in enumerate(zip(code_file, nl_file)):
            prompt = clean_string(code_line)
            completion = clean_string(nl_line)

            try:
                ws.append([prompt, completion])
            except Exception as e:
                print(f"跳过非法行 {i+1}: {e}")

            # 控制写入行数（可选）
            if max_rows and i + 1 >= max_rows:
                break

    # 保存为 Excel 文件
    wb.save(output_excel_path)
    print(f"已生成 Excel 文件: {output_excel_path}, 共 {i+1} 行。")

# 示例路径
code_file = 'Clean_PCSD/test/code'
nl_file = 'Clean_PCSD/test/nl.txt'
output_excel = 'aliPCSDData/test_500.xlsx'

# 执行函数（若只想导出前100条测试，可以加 max_rows=100）
generate_prompt_completion_excel(code_file, nl_file, output_excel,max_rows=500)
