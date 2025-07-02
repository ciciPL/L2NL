import json
import random
import numpy as np
import torch
import math
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --------------------------------------------------------------------------
# [第零部分] 环境设置 (与 llama-factory 对齐)
# --------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- 1. 配置 ---
BATCH_SIZE = 8
base_model_path = "../../model/deepseek-coder-1.3b-instruct"
lora_checkpoint_path = "../../finetune/output_train_6k_with_sentence_myMetric/checkpoint-600"
data_file_path = "../../dataset/finetune/alpacaPCSD/Racket_python_with_sentence_structure.json"

# --- [明确] 定义唯一的输出文件路径，并明确其为JSONL格式 ---
output_raw_ids_path = "../../experiment/ds-coder-sentences/Racket_python_nl_sentences_structure.jsonl"

# --- 2. 加载模型和分词器 ---
print("正在加载模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# --- [关键] 与 llama-factory 的 pad_token 行为对齐 ---
# tokenizer.pad_token = "<｜end_of_sentence｜>"
# tokenizer.pad_token_id = 32014
# tokenizer.padding_side = "left"

tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, lora_checkpoint_path)
model = model.merge_and_unload()
model.eval()
print("模型加载完成。")


# --- 3. 定义模板构建函数 (最终修正版) ---
def build_deepseek_prompt(system_prompt, instruction, input_text):
    user_content = instruction
    if input_text and input_text.strip():
        user_content += "\n" + input_text

    # 【【【关键修改】】】
    # 在system_prompt的前面加上一个换行符 `\n`
    # 这将使得分词器在自动添加BOS token后，紧接着就是这个换行符的ID (185)
    prompt_str = (
        f"\n{system_prompt}\n\n"  # <--- 在这里加上换行符
        f"User: {user_content}\n\n"
        f"Assistant:"
    )
    return prompt_str


# --- 4. 读取数据、进行推理并保存结果 ---
system_prompt_text = "You are an expert code summarization AI. Generate a SINGLE concise, one-line summary by:\n1. Primarily analyzing the provided CODE\n2. Augmenting with core context from SIMILAR code snippets"

# --- [保留] 您健壮的JSON/JSONL加载逻辑 ---
try:
    data = []
    with open(data_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            print("已成功按标准 JSON 格式读取文件。")
        except json.JSONDecodeError:
            print("标准 JSON 读取失败，尝试按 JSONL (每行一个JSON对象) 格式读取...")
            f.seek(0)
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
            print("已成功按 JSONL 格式读取文件。")
except FileNotFoundError:
    print(f"错误: 数据文件未找到，请确保 '{data_file_path}' 文件存在。")
    exit()
except Exception as e:
    print(f"错误: 读取或解析数据文件时出错 - {e}")
    exit()
if not isinstance(data, list):
    print("错误: 数据格式不正确，需要一个包含JSON对象的列表。")
    exit()

print(f"\n--- 开始从 {data_file_path} 文件进行推理，批处理大小: {BATCH_SIZE} ---")

# --- 初始化一个列表，用于之后写入文件 ---
all_results_to_save = []

num_batches = math.ceil(len(data) / BATCH_SIZE)

for i in tqdm(range(num_batches), desc="推理进度"):
    start_index = i * BATCH_SIZE
    end_index = start_index + BATCH_SIZE
    batch_data = data[start_index:end_index]

    prompts = [
        build_deepseek_prompt(
            system_prompt_text,
            record.get("instruction", ""),
            record.get("input", "")
        ) for record in batch_data
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    labels_text = [record.get("output", "") for record in batch_data]
    labels_tokenized = tokenizer(labels_text, return_tensors="pt", padding=True).to(model.device)

    generation_output = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        top_p=0.5,
        top_k=50,
        num_beams=1,
        length_penalty=1.5,
        repetition_penalty=1.5,
        do_sample=True,
        eos_token_id=[tokenizer.eos_token_id, 32021]
    )

    # ------------------ [核心修改] 收集ID和解码文本 ------------------
    # 1. 获取 predict_ids
    predict_ids_tensor = generation_output[:, inputs['input_ids'].shape[1]:]

    # 2. 【新增】解码输入和预测的文本
    # 解码输入时，不跳过特殊token，这样能看到完整的、包含BOS/padding的输入文本
    decoded_inputs = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False)
    # 解码预测时，跳过特殊token，得到干净的生成文本
    decoded_predicts = tokenizer.batch_decode(predict_ids_tensor, skip_special_tokens=True)

    decoded_labels = tokenizer.batch_decode(labels_tokenized["input_ids"], skip_special_tokens=True)

    # 3. 遍历当前批次，将所有需要的信息存入列表
    for j in range(len(batch_data)):
        all_results_to_save.append({
            "input_ids": inputs["input_ids"][j].tolist(),
            "predict_ids": predict_ids_tensor[j].tolist(),
            "label_ids": labels_tokenized["input_ids"][j].tolist(),
            "decoded_input": decoded_inputs[j],
            "decoded_predict": decoded_predicts[j].strip(),
            "decoded_label": decoded_labels[j]
        })

# --- 5. 将收集到的所有结果以JSONL格式写入文件 ---
try:
    with open(output_raw_ids_path, 'w', encoding='utf-8') as f:
        for record in all_results_to_save:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n所有记录（包含ID和解码文本）推理完成，结果已保存到: {output_raw_ids_path}")
except Exception as e:
    print(f"\n保存结果到JSONL文件时出错: {e}")
