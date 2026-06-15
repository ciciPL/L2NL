import json
from tqdm import tqdm
import argparse


def merge_jsonl_files(first_file_path, second_file_path, output_file_path):
    """
    从第二个JSONL文件中读取id和raw_code，然后在第一个JSONL文件中查找对应id的数据，
    将raw_code字段添加到第一个文件的数据中并输出到新文件。

    Args:
        first_file_path (str): 第一个JSONL文件路径（包含完整数据）
        second_file_path (str): 第二个JSONL文件路径（包含id和raw_code）
        output_file_path (str): 输出文件路径
    """

    # 第一步：读取第一个文件，建立id到完整数据的映射
    print("正在读取第一个文件...")
    first_data_dict = {}
    total_first = 0

    try:
        with open(first_file_path, 'r', encoding='utf-8') as f:
            # 计算总行数用于进度条
            total_lines = sum(1 for _ in f)

        with open(first_file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="读取第一个文件"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'idx' in data:
                        first_data_dict[data['idx']] = data
                        total_first += 1
                    else:
                        print(f"警告: 发现缺少'idx'字段的数据行")
                except json.JSONDecodeError as e:
                    print(f"警告: JSON解析失败 - {e}")
    except FileNotFoundError:
        print(f"错误: 文件 '{first_file_path}' 未找到")
        return
    except Exception as e:
        print(f"错误: 读取第一个文件时发生异常 - {e}")
        return

    print(f"第一个文件中共有 {total_first} 条数据")

    # 第二步：读取第二个文件，提取id和raw_code
    print("正在处理第二个文件...")
    merged_count = 0
    not_found_count = 0
    json_error_count = 0
    missing_field_count = 0

    try:
        with open(second_file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        # 第三步：处理第二个文件并合并数据
        with open(second_file_path, 'r', encoding='utf-8') as second_f, \
                open(output_file_path, 'w', encoding='utf-8') as output_f:

            for line in tqdm(second_f, total=total_lines, desc="合并数据"):
                line = line.strip()
                if not line:
                    continue

                try:
                    second_data = json.loads(line)

                    # 检查必需字段
                    if 'id' not in second_data:
                        missing_field_count += 1
                        print(f"警告: 缺少'id'字段")
                        continue

                    if 'raw_code' not in second_data:
                        missing_field_count += 1
                        print(f"警告: ID {second_data.get('id', 'unknown')} 缺少'raw_code'字段")
                        continue

                    data_id = second_data['id']
                    raw_code = second_data['raw_code']

                    # 在第一个文件中查找对应id的数据
                    if data_id in first_data_dict:
                        # 复制原始数据并添加raw_code字段
                        merged_data = first_data_dict[data_id].copy()
                        merged_data['raw_code'] = raw_code

                        # 写入输出文件
                        output_f.write(json.dumps(merged_data, ensure_ascii=False) + '\n')
                        merged_count += 1
                    else:
                        not_found_count += 1
                        print(f"警告: 在第一个文件中未找到ID '{data_id}' 对应的数据")

                except json.JSONDecodeError as e:
                    json_error_count += 1
                    print(f"警告: JSON解析失败 - {e}")
                except Exception as e:
                    print(f"警告: 处理数据时发生异常 - {e}")

    except FileNotFoundError:
        print(f"错误: 文件 '{second_file_path}' 未找到")
        return
    except Exception as e:
        print(f"错误: 处理文件时发生异常 - {e}")
        return

    # 输出统计信息
    print("\n=== 处理完成 ===")
    print(f"成功合并: {merged_count} 条数据")
    print(f"未找到对应数据: {not_found_count} 条")
    print(f"JSON解析错误: {json_error_count} 条")
    print(f"字段缺失: {missing_field_count} 条")
    print(f"输出文件: {output_file_path}")


def jsonl_to_json_basic(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为JSON数组格式

    Args:
        jsonl_file_path (str): 输入的JSONL文件路径
        json_file_path (str): 输出的JSON文件路径
    """
    data = []

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行JSON解析失败，已跳过: {e}")

        # 写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

        print(f"转换完成！共处理 {len(data)} 条记录")
        print(f"输出文件: {json_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{jsonl_file_path}' 未找到")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="JSONL to JSON conversion utility. "
                    "Two modes: (a) basic conversion (--input/--output); "
                    "(b) merge two jsonls on `id`, copying `raw_code` from B into A (--input/--with_raw_code/--output)."
    )
    parser.add_argument("--input", default="./data/ready_sentences_dataset/train_output6k_preds.jsonl",
                        help="Primary jsonl file")
    parser.add_argument("--output", default="./data/ready_sentences_dataset/train_output6k_preds.json",
                        help="Output json file")
    parser.add_argument("--with_raw_code", default=None,
                        help="Optional secondary jsonl providing {id, raw_code}; "
                             "when set, runs merge_jsonl_files instead of basic conversion")
    args = parser.parse_args()

    if args.with_raw_code:
        merge_jsonl_files(args.input, args.with_raw_code, args.output)
    else:
        jsonl_to_json_basic(args.input, args.output)
