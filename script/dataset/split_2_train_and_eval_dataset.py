import json
import random
import argparse
from tqdm import tqdm # 用于显示进度条

def split_dataset(input_file_path, train_file_path, val_file_path, val_ratio=0.1, shuffle=True, random_seed=42):
    """
    将一个包含JSON对象的数组的JSON文件划分为训练集和验证集 (输出为JSONL格式)。

    参数:
    input_file_path (str): 输入的JSON文件路径 (格式应为 [{}, {}, ...])。
    train_file_path (str): 输出的训练集 JSONL 文件路径。
    val_file_path (str): 输出的验证集 JSONL 文件路径。
    val_ratio (float): 验证集所占的比例 (例如 0.1 表示 10%)。
    shuffle (bool): 是否在分割前打乱数据。
    random_seed (int): 打乱数据时使用的随机种子，确保结果可复现。
    """
    print(f"正在读取原始数据文件: {input_file_path}...")
    all_data = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            # --- 改动点：一次性加载整个JSON文件 ---
            content = infile.read()
            if not content.strip():
                print("错误: 输入文件为空。")
                return
            all_data = json.loads(content) # 解析整个文件内容为一个JSON对象（这里期望是一个列表）
            # --- 改动结束 ---

        if not isinstance(all_data, list):
            print("错误: JSON文件顶层结构不是一个列表/数组。期望的格式是 [{}, {}, ...]。")
            return

    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file_path}' 未找到。")
        return
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON时出错 - {e}。请检查文件是否为有效的JSON数组格式。")
        return

    if not all_data:
        print("错误: 输入文件解析后数据为空。")
        return

    total_samples = len(all_data)
    print(f"总共读取到 {total_samples} 条数据。")

    if shuffle:
        print(f"正在打乱数据 (随机种子: {random_seed})...")
        random.seed(random_seed)
        random.shuffle(all_data)

    num_val_samples = int(total_samples * val_ratio)
    if num_val_samples == 0 and total_samples > 0 and val_ratio > 0:
        num_val_samples = 1 if total_samples > 1 else 0
    if num_val_samples >= total_samples :
        if total_samples > 1:
            num_val_samples = total_samples - 1
            print(f"警告: 验证集比例过高或数据太少，调整验证集数量为 {num_val_samples}，以确保训练集至少有一条数据。")
        else:
            num_val_samples = 0
            print(f"警告: 数据集过小（只有{total_samples}条），无法分割出验证集。所有数据将作为训练集。")

    val_data = all_data[:num_val_samples]
    train_data = all_data[num_val_samples:]

    num_train_samples = len(train_data)
    num_val_samples_actual = len(val_data)

    print(f"分割完成：训练集 {num_train_samples} 条，验证集 {num_val_samples_actual} 条。")

    print(f"正在写入训练集到 (JSONL格式): {train_file_path}...")
    with open(train_file_path, 'w', encoding='utf-8') as outfile_train:
        for item in tqdm(train_data, desc="写入训练集"):
            outfile_train.write(json.dumps(item, ensure_ascii=False) + '\n')

    if num_val_samples_actual > 0:
        print(f"正在写入验证集到 (JSONL格式): {val_file_path}...")
        with open(val_file_path, 'w', encoding='utf-8') as outfile_val:
            for item in tqdm(val_data, desc="写入验证集"):
                outfile_val.write(json.dumps(item, ensure_ascii=False) + '\n')
    else:
        print("没有验证集数据被写入。")

    print("数据划分完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="手动将 JSONL 数据集划分为训练集和验证集。")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集所占的比例，默认为 0.1 (10%)。")
    parser.add_argument("--no_shuffle", action="store_false", dest="shuffle", help="指定此项则不在分割前打乱数据。")
    parser.add_argument("--seed", type=int, default=42, help="打乱数据时使用的随机种子，默认为 42。")

    args = parser.parse_args()

    split_dataset(
        input_file_path='../../dataset/finetune/alpacaPCSD/train_6k_without_sentence.json',
        train_file_path='../../finetune/LLaMA-Factory/data/train_6k_without_sentence_split.json',
        val_file_path='../../finetune/LLaMA-Factory/data/val_6k_without_sentence.json',
        val_ratio=args.val_ratio,
        shuffle=args.shuffle,
        random_seed=args.seed
    )