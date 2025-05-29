import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from tqdm import tqdm
import os
import re
import logging

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告：vLLM 未安装。generate_summaries_vllm 函数将不可用。")

# 禁用 transformers 库的详细日志
logging.getLogger('transformers').setLevel(logging.ERROR)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# def custom_collate_fn(batch):
#
#     system_content = """You are a code summarizer. Generate ONE-LINE summaries(<=25 words) with this format:
#
#     genSummary:\tsummary
#
#
#
#     Rules:
#
#     - Omit explanations
#
#     - Never repeat doc content
#
#     """
#
#     messages = []
#
#     indices = []
#
#     for item in batch:
#
#     prompt = f"""CODE: {item['code']}
#
#     DOC: {item['doc']}"""
#
#     message = [
#
#     {"role": "system", "content": system_content},
#
#     {"role": "user", "content": prompt}
#
#     ]
#
#     messages.append(message)
#
#     indices.append(item['index'])
#
#     return messages, indices
# --- 自定义数据集 (保持不变) ---
class CodeSummaryDataset(Dataset):
    def __init__(self, doc_file, code_file):
        with open(doc_file, 'r', encoding='utf-8') as doc_f, \
                open(code_file, 'r', encoding='utf-8') as code_f:
            doc_lines = doc_f.readlines()
            code_lines = code_f.readlines()
        self.data = []
        for index in range(len(doc_lines)):
            doc_line = re.sub(r'^\d+:\s*', '', doc_lines[index]).strip()
            # 确保分割后至少有两个部分，并取第二个
            parts = doc_line.split('\t')
            if len(parts) > 1:
                doc_line = parts[1]
            else:
                doc_line = parts[0]  # 或者设置为 "" 或其他默认值

            code_line = re.sub(r'^\d+:\s*', '', code_lines[index]).strip()
            self.data.append({
                'index': index + 1,
                'doc': doc_line,
                'code': code_line
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- 自定义 Collate 函数 (保持不变) ---
def custom_collate_fn(batch):
    system_content = """You are an expert code summarization AI.
Your task is to generate a concise, ONE-LINE summary for the provided CODE and its DOC.

Instructions:
1. The summary MUST be a single line.
3. The summary MUST accurately reflect the core functionality described in the CODE and DOC.
4. Do NOT include any explanations or conversational phrases.
5. Do NOT repeat content from the DOC.
6. Your response MUST strictly follow this format: <summary>YOUR_GENERATED_SUMMARY_HERE<end>

Example:
CODE: def add(a, b): return a + b
DOC: This function adds two numbers and return the result.
Your expected output: <summary>This function adds two numbers.<end>
"""
    messages = []
    indices = []
    for item in batch:
        # 注意：在实际发送给模型的 "user" prompt 中，我们不包含 <summary> 和 <end>
        # 因为那是模型需要生成的。
        prompt = f"""CODE:
{item['code']}

DOC:
{item['doc']}

Generate the summary based on the instructions above:"""  # 引导模型开始生成

        message = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        messages.append(message)
        indices.append(item['index'])
    return messages, indices


# --- Transformers 推理函数 ---
def generate_summaries_transformers(model_name, dataset, batch_size, output_file):
    print("\n--- 开始使用 Transformers 进行推理 ---")

    if not torch.cuda.is_available():
        print("错误：Transformers 推理需要 CUDA 支持。")
        return

    # 加载模型和分词器
    print(f"正在加载 Transformers 模型: {model_name}...")
    start_m = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 使用 bfloat16 可能更快且显存占用更低
        device_map="auto",
        attn_implementation="flash_attention_2",  # 确保已安装 flash-attn
        trust_remote_code=True
    )
    end_m = time.perf_counter()
    print(f"Transformers 模型加载耗时: {end_m - start_m:.6f} 秒")

    start_t = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = 'left'  # 对于批量生成，通常将填充放在左侧
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token
    end_t = time.perf_counter()
    print(f"Tokenizer 加载耗时: {end_t - start_t:.6f} 秒")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=os.cpu_count() // 2, pin_memory=True
    )

    with open(output_file, 'w', encoding='utf-8') as ref_file:
        for batch_messages, batch_indices in tqdm(dataloader, desc="Transformers 生成进度"):
            try:
                # 准备输入
                batch_texts = [
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ) for messages in batch_messages
                ]
                model_inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096  # 增加截断
                ).to(model.device)

                # 生成
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=40,
                    min_new_tokens=10,
                    temperature=0.1,
                    repetition_penalty=1.05,
                    num_beams=1,  # 使用 num_beams=1 (Greedy) 或 do_sample=True
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )

                # 解码并处理
                input_lengths = model_inputs.input_ids.shape[1]
                decoded_outputs = tokenizer.batch_decode(generated_ids[:, input_lengths:], skip_special_tokens=True)

                for index, text in zip(batch_indices, decoded_outputs):
                    response = re.sub(r'^genSummary:\s*', '', text.strip()).strip()
                    ref_file.write(f"{index}: {response}\n")
                    # print(f"Transformers 第 {index} 条: {response}") # 可以取消注释来实时查看

            except Exception as e:
                print(f"批次 {batch_indices} 处理失败 (Transformers): {e}")
                for index in batch_indices:
                    ref_file.write(f"{index}: 生成失败\n")

    print(f"Transformers 推理完成，结果已写入: {output_file}")


# --- vLLM 推理函数 ---
def generate_summaries_vllm(model_name, dataset, batch_size, output_file, tensor_parallel_size=1):
    print("\n--- 开始使用 vLLM 进行推理 ---")
    # ... (前面的加载代码不变) ...

    # 初始化 vLLM 模型
    print(f"正在加载 vLLM 模型 ({model_name})...")
    start_m = time.perf_counter()
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size
    )
    end_m = time.perf_counter()
    print(f"vLLM 模型加载耗时: {end_m - start_m:.6f} 秒")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=custom_collate_fn, num_workers=os.cpu_count() // 2, pin_memory=True
    )

    sampling_params = SamplingParams(
        temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=40, min_tokens=15
    )

    with open(output_file, 'w', encoding='utf-8') as ref_file:
        for batch_messages, batch_indices in tqdm(dataloader, desc="vLLM 生成进度"):
            try:
                batch_prompts_with_ids = []
                for i, messages in enumerate(batch_messages):
                    prompt_str = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # 使用原始数据集的 index 作为 request_id
                    # vLLM.generate API 接受 (prompt, sampling_params, request_id) 的形式
                    # 但更通用的做法是在 generate_async 或 add_request 时指定
                    # 对于同步的 llm.generate(prompts_list, ...), 它会自动分配0,1,2...
                    # 但这些 ID 是相对于 vLLM 内部的，如果我们想自己控制，
                    # 我们需要一种方法在返回时能够对应上。
                    #
                    # vLLM 的 llm.generate 实际上并不直接接受 request_id 作为列表中每个元素的参数。
                    # 它返回的 RequestOutput 中的 request_id 是 vLLM 自己管理的。
                    #
                    # 鉴于 vLLM 的 request_id 是全局累积的，
                    # 并且您发现它与批内索引不匹配，
                    # 最好的方式是，在收到 batch_outputs 后，
                    # 假设 batch_outputs 的顺序与 batch_prompts 的输入顺序严格对应。
                    # 然后，我们可以使用 batch_indices 来正确地写入文件。

                batch_prompts = [
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ) for messages in batch_messages
                ]

                batch_outputs = llm.generate(batch_prompts, sampling_params)

                if len(batch_outputs) != len(batch_indices):
                    print(f"警告: 输入 {len(batch_indices)} 个, 但 vLLM 只返回 {len(batch_outputs)} 个输出。")

                # 既然 request_id 是全局累积的，我们不能依赖它和批内 enumerate 的 i 匹配。
                # 但我们可以安全地假设 batch_outputs[j] 对应 batch_indices[j] 的输入。
                for i in range(len(batch_outputs)):  # 迭代 vLLM 返回的实际输出数量
                    if i < len(batch_indices):  # 确保我们不会因为输出少而越界
                        output = batch_outputs[i]
                        original_data_index = batch_indices[i]  # 获取对应的原始数据索引

                        # (可选) 打印vLLM的request_id，仅供观察
                        # print(f"  --> 调试: vLLM request_id: {output.request_id} for original_data_index: {original_data_index}")

                        generated_text = output.outputs[0].text.strip()
                        match = re.search(r"<summary>(.*?)<end>", generated_text)
                        if match:
                            summary = match.group(1)
                        else:
                            # 处理模型未按预期格式输出的情况
                            print("模型输出格式不符合预期:", generated_text)
                            summary = generated_text  # 或者其他回退逻辑
                        ref_file.write(f"{original_data_index}: {summary}\n")
                    else:
                        # 这种情况理论上不应该发生，因为我们迭代 len(batch_outputs)
                        pass


            except Exception as e:
                print(f"批次 {batch_indices} 处理失败 (vLLM): {e}")
                # 当批次失败时，为该批次的所有索引写入失败标记
                for index_in_batch_failure in batch_indices:
                    ref_file.write(f"{index_in_batch_failure}: 生成失败 (批次异常)\n")

    print(f"vLLM 推理完成，结果已写入: {output_file}")


# --- 主函数 ---
def main():
    # 预热GPU
    if torch.cuda.is_available():
        torch.zeros(1).cuda()

    # --- 配置 ---
    model_name = "model/ByteDance-Seed/Seed-Coder-8B-Instruct"  # 或者本地路径
    doc_file = 'finetune/dataset/Clean_PCSD/train/ref.txt'
    code_file = 'finetune/dataset/Clean_PCSD/train/code'
    output_file_transformers = 'finetune/dataset/Clean_PCSD/train/PCSD_ref_gen_transformers.txt'
    output_file_vllm = 'finetune/dataset/Clean_PCSD/train/PCSD_ref_gen_vllm.txt'
    batch_size = 16  # 根据显存调整
    vllm_gpus = 2  # 设置 vLLM 使用的 GPU 数量

    # --- 创建数据集 ---
    print("正在创建数据集...")
    dataset = CodeSummaryDataset(doc_file, code_file)
    print(f"数据集创建完成，共 {len(dataset)} 条数据。")

    # --- 选择要运行的函数 ---

    # 选项 1: 运行 Transformers 推理
    # generate_summaries_transformers(model_name, dataset, batch_size, output_file_transformers)

    # 选项 2: 运行 vLLM 推理 (如果已安装)
    generate_summaries_vllm(model_name, dataset, batch_size, output_file_vllm, tensor_parallel_size=vllm_gpus)

    print("\n所有选择的推理任务已完成。")


if __name__ == '__main__':
    main()
