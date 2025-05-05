import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# 定义和训练时完全一样的 EOT token
EOT_TOKEN = "<|EOT|>"

# 定义和训练时完全一样的 prompt 构建函数
def build_instruction_prompt(instruction: str):
    # 确保前后的空格处理和训练时一致（训练代码中 .lstrip() 暗示开头的换行可能被移除）
    return '''
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

# --- 模型和 Tokenizer 加载 ---
model_name = '../../../model/deepseek-ai/deepseek-coder-1.3b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left" # 推理时做 batch generation 通常用 left padding

# 检查并设置 pad_token (DeepSeek Coder 通常有，但确认下)
if tokenizer.pad_token is None:
    print("警告: Tokenizer 没有 pad_token。将其设置为 eos_token。")
    tokenizer.pad_token = tokenizer.eos_token # 或者确认 DeepSeek 是否有特定的 pad token id

# 检查 EOT token 是否在词汇表中 (如果不在，理论上训练时就该加并resize)
if EOT_TOKEN not in tokenizer.vocab:
     print(f"警告: {EOT_TOKEN} 不在 tokenizer 词汇表中。")
     # 通常不需要在推理时添加，除非训练时遗漏了关键步骤

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # 如果可用且有效，保留
    trust_remote_code=True)

# 加载 PEFT adapter
model = PeftModel.from_pretrained(model, '../../../finetune/ds-coder/output_dir_2th', device_map="auto", trust_remote_code=True)
model.eval() # 设置为评估模式

# 获取训练时使用的 EOT token 的 ID
eot_token_id = tokenizer.convert_tokens_to_ids(EOT_TOKEN)
print(f"使用 EOT token ID ({eot_token_id}) 作为生成停止符。")

def generate_summary_batch(code_list):
    """使用正确的训练 Prompt 格式生成摘要"""
    # 1. 构建放入模板 *内部* 的指令内容
    instructions = [
        f"Provide a concise summary of the following code:\nCode: {code}"
        for code in code_list
    ]
    # 2. 使用训练时的函数构建最终的完整 Prompt
    prompts = [
        build_instruction_prompt(instruction)
        for instruction in instructions
    ]

    # 3. Tokenize 输入
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=450).to(model.device) # max_length 留出生成空间

    # 4. 生成参数设置
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,         # 摘要本身的最大长度
        temperature=0.1,           # 低温确保输出稳定性
        num_beams=1,               # 基本等同于 Greedy Search
        do_sample=False,           # 不进行采样
        repetition_penalty=1.1,    # 如果有助于控制重复，保留
        eos_token_id=eot_token_id  # ***关键：使用训练时的 EOT token ID 来停止生成***
        # eos_token_id=[eot_token_id, tokenizer.eos_token_id] # 或者包含默认的eos token id，如果可能的话
    )

    # 5. 解码 (只解码新生成的部分)
    input_ids_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, input_ids_len:] # 切片，只取生成的部分
    # 使用 batch_decode 并跳过特殊 token (包括 <|EOT|>)
    summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # 清理可能残留的空格
    return [s.strip() for s in summaries]


def process_file(input_file, output_file, batch_size=4):
    """
    从输入文件读取代码片段，批量生成摘要，并将结果写入输出文件。
    包含进度条和更详细的错误处理。

    Args:
        input_file (str): 输入文件路径 (格式: "索引: 代码" 每行)。
        output_file (str): 输出文件路径。
        batch_size (int): 每个批次处理的代码片段数量。
    """
    print("-" * 50)
    print(f"开始处理文件: {input_file}")
    print(f"结果将保存至: {output_file}")
    print(f"使用批处理大小: {batch_size}")
    print("-" * 50)

    total_lines = 0
    # 首先尝试计算总行数，并处理文件不存在的情况
    try:
        with open(input_file, 'r', encoding='utf-8') as f_count:
            total_lines = sum(1 for _ in f_count)
        print(f"文件总行数: {total_lines}")
        if total_lines == 0:
            print("警告: 输入文件为空，无需处理。")
            # 创建一个空的输出文件或直接返回
            open(output_file, 'w').close()
            return
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_file}'")
        return # 文件不存在，无法继续
    except Exception as e:
        print(f"错误: 读取文件行数时发生异常: {e}")
        traceback.print_exc()
        return # 其他读取错误，无法继续

    # 使用 try-with-resources 确保文件正确关闭
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            # 初始化进度条
            pbar = tqdm(total=total_lines, desc="处理进度", unit="行", ncols=100)

            batch = [] # 存储当前批次的 (索引, 代码) 元组
            lines_processed_in_batch = 0 # 当前批次已处理的有效行数计数器

            for line_num, line in enumerate(infile):
                line = line.strip() # 去除首尾空白

                # 跳过空行
                if not line:
                    pbar.update(1) # 即使是空行，也更新进度条总数
                    continue

                # 检查并分割行
                if ':' in line:
                    try:
                        index, code = line.split(':', 1)
                        index = index.strip()
                        code = code.strip()

                        # 检查分割后索引或代码是否为空
                        if not index or not code:
                            pbar.write(f"警告: 跳过格式错误行 {line_num + 1} (索引或代码为空): '{line[:100]}...'") # pbar.write 避免打乱进度条
                            outfile.write(f"{index if index else '无索引'}: ❌ 格式错误 (索引或代码为空)\n")
                            pbar.update(1)
                            continue # 跳到下一行

                        # 添加到批次
                        batch.append((index, code))
                        lines_processed_in_batch += 1

                    except ValueError: # split 可能因多个 ':' 出错，虽然 split(':'. 1) 避免了大部分情况
                        pbar.write(f"警告: 跳过格式错误行 {line_num + 1} (分割错误): '{line[:100]}...'")
                        outfile.write(f"行_{line_num+1}: ❌ 格式错误 (分割问题)\n")
                        pbar.update(1)
                        continue
                else:
                    # 行中缺少 ':' 分隔符
                    pbar.write(f"警告: 跳过格式错误行 {line_num + 1} (缺少 ':'): '{line[:100]}...'")
                    outfile.write(f"行_{line_num+1}: ❌ 格式错误 (缺少 ':')\n")
                    pbar.update(1)
                    continue

                # 当批次达到指定大小时，处理批次
                if lines_processed_in_batch >= batch_size:
                    indices, codes = zip(*batch) # 解包索引和代码列表
                    try:
                        # **调用修正后的批量生成函数**
                        summaries = generate_summary_batch(list(codes))

                        # 检查返回的摘要数量是否与输入数量一致
                        if len(summaries) != len(indices):
                            pbar.write(f"错误: 批次处理返回的摘要数量 ({len(summaries)}) 与输入数量 ({len(indices)}) 不匹配。")
                            for idx in indices:
                                outfile.write(f"{idx}: ❌ 批次处理错误 (数量不匹配)\n")
                        else:
                            # 正常写入结果
                            for idx, summary in zip(indices, summaries):
                                outfile.write(f"{idx}: {summary}\n")

                    except Exception as e:
                        # 捕获 generate_summary_batch 中的任何异常
                        pbar.write(f"\n--- 错误: 处理批次时发生异常 (大约在行 {line_num + 1 - batch_size + 1} 附近) ---")
                        pbar.write(f"异常类型: {type(e).__name__}")
                        pbar.write(f"异常信息: {e}")
                        # 打印详细的回溯信息到控制台
                        traceback.print_exc()
                        pbar.write(f"该批次部分代码 (前50字符): {[c[:50]+'...' for c in codes]}")
                        pbar.write("--- 错误结束 ---")

                        # 为该批次中的所有条目写入错误标记
                        for idx in indices:
                            outfile.write(f"{idx}: ❌ 批次处理异常 (详情请查看控制台日志)\n")

                    # 清空批次并重置计数器，为下一个批次做准备
                    batch = []
                    lines_processed_in_batch = 0
                    # 更新进度条 (增加刚处理完的批次大小)
                    # 注意：这里更新的值应该是实际处理的有效行数，即 batch_size
                    # 如果上面有 continue 跳过的行，pbar 已经单独更新过了
                    pbar.update(batch_size)


            # 循环结束后，处理可能剩余的不满一个批次的数据
            if batch:
                pbar.write(f"\n处理最后剩余的 {len(batch)} 个条目...")
                indices, codes = zip(*batch)
                try:
                    summaries = generate_summary_batch(list(codes))
                    if len(summaries) != len(indices):
                         pbar.write(f"错误: 最后批次处理返回的摘要数量 ({len(summaries)}) 与输入数量 ({len(indices)}) 不匹配。")
                         for idx in indices:
                             outfile.write(f"{idx}: ❌ 最后批次处理错误 (数量不匹配)\n")
                    else:
                        for idx, summary in zip(indices, summaries):
                            outfile.write(f"{idx}: {summary}\n")

                except Exception as e:
                    pbar.write(f"\n--- 错误: 处理最后批次时发生异常 ---")
                    pbar.write(f"异常类型: {type(e).__name__}")
                    pbar.write(f"异常信息: {e}")
                    traceback.print_exc()
                    pbar.write(f"该批次部分代码 (前50字符): {[c[:50]+'...' for c in codes]}")
                    pbar.write("--- 错误结束 ---")
                    for idx in indices:
                        outfile.write(f"{idx}: ❌ 最后批次处理异常 (详情请查看控制台日志)\n")

                # 更新进度条 (增加最后这个批次的大小)
                pbar.update(len(batch))

            # 确保进度条最终达到总数 (处理一些极端情况，比如文件末尾空行等)
            if pbar.n < total_lines:
                 pbar.update(total_lines - pbar.n)
            pbar.close() # 关闭进度条

    except IOError as e:
        print(f"错误: 读写文件时发生 IO 错误: {e}")
        traceback.print_exc()
    except Exception as e:
        # 捕获其他意外错误
        print(f"错误: 处理文件时发生未预料的异常: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    input_file = "../rkt_result/rkt_2_python_40510.txt"
    # 建议使用新的输出文件名以区分结果
    output_file = "../rkt_result/rkt_2_python_2_NL_sft_ds_penalty1.1.txt"
    batch_size = 4

    print("使用修正后的 Prompt 开始代码摘要生成...")
    process_file(input_file, output_file, batch_size)
    print(f"摘要生成完成！结果已保存至 {output_file}")