import json

# from dataset.bertData import code_lines


# def mergeMulti2single(input_file,output_file):
#
#     with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
#         buffer = ""
#         for line in fin:
#             buffer += line.strip()
#             try:
#                 obj = json.loads(buffer)
#                 fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
#                 buffer = ""
#             except json.JSONDecodeError:
#                 continue  # 还没读完一条数据，继续累加
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
#         nl_file = os.path.join(base_dir, split, 'nl')
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
#                 # 构建 DeepSeek Coder 训练所需的 JSON 格式
#                 entry = {
#                     "instruction": "Provide a concise summary of the following code:"+code,
#                     "output": nl
#                 }
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
# base_dir = '../dataset/Clean_PCSD'  # 根据实际路径调整
# output_dir = '../dataset/PCSD'
#
# process_pcsd_dataset(base_dir, output_dir)

# input_file ='../dataset/PCSD/train_pcsd.jsonl'
# output_file = '../dataset/PCSD/train_pcsd_clean.jsonl'
# mergeMulti2single(input_file, output_file)


# 混合训练集转化为ds所需的json格式
baseRef_dir = '../dataset/ali_PCSD/fixedRef_clean.txt'  # 根据实际路径调整
baseCode_dir = '../dataset/ali_PCSD/alicode_fixed.txt'
output_dir = '../dataset/PCSD/train_fixed.jsonl'


# ids = []
# with open(baseRef_dir, 'r', encoding='utf-8') as f_nl:
#     for line in f_nl:
#         ids.append(line.strip().split(':')[0])
# with open(baseCode_dir, 'r', encoding='utf-8') as f_code, \
#         open('../dataset/ali_PCSD/alicode_fixed.txt', 'w', encoding='utf-8') as f_out:
#     for line in f_code:
#         index = line.strip().split(':')[0]
#         if index in ids:
#             f_out.write(line)


def process_fix_dataset(baseRef_dir, baseCode_dir, output_dir):
    # 处理数据
    with open(baseCode_dir, 'r', encoding='utf-8') as f_code, \
            open(baseRef_dir, 'r', encoding='utf-8') as f_nl, \
            open(output_dir, 'w', encoding='utf-8') as f_out:
        for code, nl in zip(f_code, f_nl):
            code = code.strip().split(':')[1]
            nl = nl.strip().split(':')[1]
            # Provide a concise summary of the following code:
            # 构建 DeepSeek Coder 训练所需的 JSON 格式
            entry = {
                "instruction": "Generate a ONE-LINE summary (≤25 words) for this code:" + '\nCode:' + code + '\nSummary:',
                "output": nl
            }

            # 写入 JSONL 文件
            f_out.write(json.dumps(entry) + '\n')
process_fix_dataset(baseRef_dir, baseCode_dir, output_dir)

