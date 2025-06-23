import os

#该文件用于把真实摘要和合成摘要合并

# Define filenames
file_real_cleaned = "../../dataset/finetune/Clean_PCSD/train/extract_ref.txt"
file_synthetic_cleaned = "../../dataset/finetune/Clean_PCSD/train/extract_PCSD_ref_gen_vllm_prompt2th_clean.txt"
file_synthetic_uncleaned = "../../dataset/finetune/Clean_PCSD/train/PCSD_ref_gen_vllm_prompt2th_clean.txt" # User stated this is uncleaned
output_file = "fixed_20872_36977_3.6v6.4.txt"

# --- Helper function to read a file into a dictionary ---
def read_file_to_dict(filepath):
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"警告: 文件 '{filepath}' 不存在。")
        return data_dict
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    index_str, summary = line.split(':', 1)
                    index = int(index_str)
                    data_dict[index] = line # Store the whole line
                except ValueError:
                    print(f"警告: 在文件 '{filepath}' 中跳过格式错误的行: {line}")
                except Exception as e:
                    print(f"警告: 处理文件 '{filepath}' 行 '{line}' 时发生错误: {e}")
    except Exception as e:
        print(f"错误: 读取文件 '{filepath}' 失败: {e}")
    return data_dict

# --- Main logic ---
final_data = {}
added_indices = set()

# 1. Read from extract_ref.txt (cleaned real summaries)
print(f"步骤 1: 读取 '{file_real_cleaned}'...")
real_cleaned_data = read_file_to_dict(file_real_cleaned)
for index, line_content in real_cleaned_data.items():
    final_data[index] = line_content
    added_indices.add(index)
print(f"从 '{file_real_cleaned}' 添加了 {len(real_cleaned_data)} 条目。")
print(f"当前总条目数: {len(final_data)}。已记录索引数: {len(added_indices)}。")


# 2. Read from extract_PCSD_ref_gen_vllm_prompt2th_clean.txt (cleaned synthetic summaries)
print(f"\n步骤 2: 读取 '{file_synthetic_cleaned}'...")
synthetic_cleaned_data = read_file_to_dict(file_synthetic_cleaned)
count_step2 = 0
for index, line_content in synthetic_cleaned_data.items():
    if index not in added_indices:
        final_data[index] = line_content
        added_indices.add(index)
        count_step2 += 1
print(f"从 '{file_synthetic_cleaned}' 添加了 {count_step2} 个新条目。")
print(f"当前总条目数: {len(final_data)}。已记录索引数: {len(added_indices)}。")

# 3. Check for missing indices (1-57849) and fill from uncleaned synthetic
print(f"\n步骤 3: 检查索引 1-57849 的连续性并从 '{file_synthetic_uncleaned}' 填充...")
max_expected_index = 57849
all_expected_indices = set(range(1, max_expected_index + 1))

missing_indices = all_expected_indices - added_indices
print(f"在 1-{max_expected_index} 范围内找到 {len(missing_indices)} 个缺失的索引。")

count_step3 = 0
if missing_indices:
    print(f"尝试从 '{file_synthetic_uncleaned}' 填充缺失的索引...")
    synthetic_uncleaned_data = read_file_to_dict(file_synthetic_uncleaned)
    if not synthetic_uncleaned_data:
        print(f"警告: '{file_synthetic_uncleaned}' 为空或无法读取，无法填充缺失的索引。")
    else:
        for index_to_fill in sorted(list(missing_indices)): # Process in order for clarity
            if index_to_fill in synthetic_uncleaned_data:
                final_data[index_to_fill] = synthetic_uncleaned_data[index_to_fill]
                added_indices.add(index_to_fill) # Though already covered by final_data keys
                count_step3 +=1
            # else:
                # print(f"注意: 索引 {index_to_fill} 在 '{file_synthetic_uncleaned}' 中也未找到。")
        print(f"从 '{file_synthetic_uncleaned}' 填充了 {count_step3} 个缺失的条目。")
else:
    print("在 1-57849 范围内没有缺失的索引。")

print(f"当前总条目数: {len(final_data)}。")

# 4. Write to fixed_20872_36977_3.6v6.4.txt
print(f"\n步骤 4: 将结果写入 '{output_file}'...")
if not final_data:
    print("没有数据可写入。输出文件将为空。")
    with open(output_file, 'w', encoding='utf-8') as f:
        pass # Create an empty file
else:
    # Sort by index before writing
    sorted_indices = sorted(final_data.keys())
    with open(output_file, 'w', encoding='utf-8') as f:
        for index in sorted_indices:
            f.write(final_data[index] + '\n')
    print(f"成功将 {len(final_data)} 条条目写入 '{output_file}'。")
    print(f"最终文件中最小索引: {min(final_data.keys()) if final_data else 'N/A'}")
    print(f"最终文件中最大索引: {max(final_data.keys()) if final_data else 'N/A'}")

# Final check on index range in output
output_indices = set(final_data.keys())
gaps_in_output = all_expected_indices - output_indices
if gaps_in_output:
    print(f"\n警告: 最终输出 '{output_file}' 在 1-{max_expected_index} 范围内仍有 {len(gaps_in_output)} 个缺失的索引。")
    # print(f"缺失的索引示例 (最多10个): {sorted(list(gaps_in_output))[:10]}")
else:
    print(f"\n最终输出 '{output_file}' 完整覆盖了 1-{max_expected_index} 的索引范围。")

print("\n任务完成。")