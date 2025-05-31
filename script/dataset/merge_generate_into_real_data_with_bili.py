import os
import math


# --- 辅助函数：读取文件到字典 ---
def read_file_to_dict(filepath):
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"警告: 文件 '{filepath}' 不存在。")
        return data_dict
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):  # 添加行号便于错误定位
                line = line.strip()
                if not line:
                    continue
                try:
                    index_str, summary = line.split(':', 1)
                    index = int(index_str)
                    data_dict[index] = line  # 存储整行数据
                except ValueError:
                    print(
                        f"警告: 在文件 '{filepath}' 第 {line_number} 行跳过格式错误的行 (无法分割索引或转换索引为整数): {line}")
                except Exception as e:
                    print(f"警告: 处理文件 '{filepath}' 第 {line_number} 行 '{line}' 时发生错误: {e}")
    except Exception as e:
        print(f"错误: 读取文件 '{filepath}' 失败: {e}")
    return data_dict


# --- 主要处理函数 ---
# max_total_indices_target: int = 57849 # 此参数在“有比例就不强制索引覆盖”的逻辑下不再用于强制填充
def create_merged_summaries(ratio_real_input: int, ratio_synthetic_input: int, file_real_cleaned: str,
                            file_synthetic_cleaned: str, file_synthetic_uncleaned: str):
    print(f"输入参数: 真实摘要比例份数={ratio_real_input}, 合成摘要比例份数={ratio_synthetic_input}")
    print(f"真实摘要文件: {file_real_cleaned}")
    print(f"合成清洗摘要文件: {file_synthetic_cleaned}")
    print(f"合成未清洗摘要文件: {file_synthetic_uncleaned}")
    # print(f"原目标最大索引覆盖 (仅供参考，不强制填充): {max_total_indices_target}")

    # --- 1. 加载数据 ---
    print("\n--- 步骤 1: 加载数据 ---")
    real_data_dict = read_file_to_dict(file_real_cleaned)
    synth_cleaned_dict = read_file_to_dict(file_synthetic_cleaned)
    synth_uncleaned_dict = read_file_to_dict(file_synthetic_uncleaned)

    final_data = {}
    added_indices = set()  # 用于记录已添加到 final_data 中的索引

    # --- 2. 处理真实摘要 (规则 1) ---
    print("\n--- 步骤 2: 处理真实摘要 ---")
    for index, line_content in real_data_dict.items():
        final_data[index] = line_content
        added_indices.add(index)
    num_real_actual = len(real_data_dict)
    print(f"从 '{file_real_cleaned}' 添加了 {num_real_actual} 条真实摘要。")

    # 计算目标合成摘要数量 (规则 1 的后半部分)
    if num_real_actual == 0 and ratio_real_input > 0:
        print("警告: 真实摘要数量为0，无法根据其计算合成摘要的目标数量。合成摘要将不会按比例添加。")
        num_synthetic_target = 0
    elif ratio_real_input <= 0:  # 避免除以零或无意义的比例部分
        print(f"警告: 输入的真实摘要比例份数 ({ratio_real_input}) 无效（应为正整数）。合成摘要将不会按比例添加。")
        num_synthetic_target = 0
    else:
        num_synthetic_target = int(round((num_real_actual / ratio_real_input) * ratio_synthetic_input))
    print(
        f"基于比例 {ratio_real_input}:{ratio_synthetic_input} 和 {num_real_actual} 条真实摘要，目标合成摘要数量: {num_synthetic_target}")

    # --- 3. 准备候选合成摘要池 (用于按比例选择) ---
    print("\n--- 步骤 3: 准备候选合成摘要池 (用于按比例选择) ---")
    candidate_synth_for_ratio = []

    temp_indices_in_candidate_pool = set()
    for idx, line in sorted(synth_cleaned_dict.items()):
        if idx not in added_indices:
            candidate_synth_for_ratio.append({'index': idx, 'line': line, 'source': 'cleaned'})
            temp_indices_in_candidate_pool.add(idx)

    for idx, line in sorted(synth_uncleaned_dict.items()):
        if idx not in added_indices and idx not in temp_indices_in_candidate_pool:
            candidate_synth_for_ratio.append({'index': idx, 'line': line, 'source': 'uncleaned'})

    num_available_for_ratio = len(candidate_synth_for_ratio)
    print(f"总共找到 {num_available_for_ratio} 条独立候选合成摘要 (已按索引排序，优先来自清洗数据)。")

    # --- 4. 根据比例选择合成摘要 (规则 2, 3, 4) ---
    print("\n--- 步骤 4: 根据比例选择合成摘要 ---")
    if num_available_for_ratio == 0 and num_synthetic_target > 0:
        print(f"错误: 没有可用的合成摘要来满足 {num_synthetic_target} 的目标数量。无法生成文件。")
        return None

    num_synthetic_selected_for_ratio = 0
    for item in candidate_synth_for_ratio:
        if num_synthetic_selected_for_ratio < num_synthetic_target:
            idx = item['index']
            if idx not in added_indices:  # 理论上此条件在构建candidate_synth_for_ratio时已保证
                final_data[idx] = item['line']
                added_indices.add(idx)
                num_synthetic_selected_for_ratio += 1
        else:
            break

    print(f"根据比例和优先级，已选择 {num_synthetic_selected_for_ratio} / {num_synthetic_target} (目标) 条合成摘要。")
    if num_synthetic_selected_for_ratio < num_synthetic_target and num_available_for_ratio > 0:
        print(
            f"注意: 可选的合成摘要数量 ({num_synthetic_selected_for_ratio}) 少于目标数量 ({num_synthetic_target})。已选择所有可用的。")

    # --- 步骤 5 (原索引连续性检查与缺失填充) 已根据用户指示移除，因为“有比例就不一定要索引覆盖” ---
    print("\n--- 步骤 5: 按比例选择完成，不进行强制索引覆盖填充 ---")
    print(f"当前总条目数: {len(final_data)}。已记录索引总数: {len(added_indices)}。")

    # --- 6. 生成输出文件名和统计数据 ---
    print("\n--- 步骤 6: 生成输出文件名和统计数据 ---")
    count_real_final = 0
    count_synthetic_final = 0

    for idx_final in final_data.keys():
        if idx_final in real_data_dict:
            count_real_final += 1
        else:
            count_synthetic_final += 1

    print(f"最终文件中真实摘要数量: {count_real_final}")
    print(f"最终文件中合成摘要数量: {count_synthetic_final}")
    print(f"最终文件总条目: {len(final_data)}")

    total_for_ratio_calc = count_real_final + count_synthetic_final
    if total_for_ratio_calc == 0:
        print("警告: 最终数据为空，无法计算实际比例。")
        achieved_ratio_str = "0.0v0.0"
    else:
        x_achieved = (count_real_final / total_for_ratio_calc) * 10
        y_achieved = (count_synthetic_final / total_for_ratio_calc) * 10
        achieved_ratio_str = f"{x_achieved:.1f}v{y_achieved:.1f}"

    output_filename = f"fixed_{count_real_final}_{count_synthetic_final}_{achieved_ratio_str}.txt"
    print(f"实际达成比例 (缩放至和为10): {achieved_ratio_str}")
    print(f"输出文件名将是: {output_filename}")

    # --- 7. 写入输出文件 ---
    print(f"\n--- 步骤 7: 将结果写入 '{output_filename}' ---")
    if not final_data:
        print("没有数据可写入。输出文件将为空，但仍会创建。")
        with open(output_filename, 'w', encoding='utf-8') as f:
            pass
    else:
        sorted_indices_output = sorted(final_data.keys())
        with open(output_filename, 'w', encoding='utf-8') as f:
            for index_output in sorted_indices_output:
                f.write(final_data[index_output] + '\n')
        print(f"成功将 {len(final_data)} 条条目写入 '{output_filename}'。")
        min_idx_final = min(final_data.keys()) if final_data else 'N/A'
        max_idx_final = max(final_data.keys()) if final_data else 'N/A'
        print(f"最终文件中最小索引: {min_idx_final}")
        print(f"最终文件中最大索引: {max_idx_final}")
        print(f"最终文件覆盖的索引范围大致为 {min_idx_final}-{max_idx_final} (具体取决于数据分布)。")

    print("\n任务完成。")
    return output_filename


# --- 使用示例 ---
if __name__ == '__main__':

    dummy_real_file = "extract_ref.txt"
    dummy_synth_c_file = "extract_PCSD_ref_gen_vllm_prompt2th_clean.txt"
    dummy_synth_u_file = "PCSD_ref_gen_vllm_prompt2th_clean.txt"

    output_file_tc1 = create_merged_summaries(
        ratio_real_input=6,
        ratio_synthetic_input=4,
        file_real_cleaned=dummy_real_file,
        file_synthetic_cleaned=dummy_synth_c_file,
        file_synthetic_uncleaned=dummy_synth_u_file
        # max_total_indices_target 被注释掉了，因为不再强制填充
    )
    print(f"--- 测试用例 1 完成。输出文件: {output_file_tc1 if output_file_tc1 else '未生成'} ---\n")
