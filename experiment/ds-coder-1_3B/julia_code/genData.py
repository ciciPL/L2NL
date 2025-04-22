import torch  
from torch.utils.data import Dataset, DataLoader  
from transformers import AutoModelForCausalLM, AutoTokenizer  
import time  
from tqdm import tqdm  
import os  
import re  
import logging  

# 禁用 transformers 库的详细日志  
logging.getLogger('transformers').setLevel(logging.ERROR)  

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  

# 自定义数据集  
class CodeSummaryDataset(Dataset):  
    def __init__(self, doc_file, code_file):  
        # 读取文件  
        with open(doc_file, 'r', encoding='utf-8') as doc_f, \
                open(code_file, 'r', encoding='utf-8') as code_f:  
            doc_lines = doc_f.readlines()  
            code_lines = code_f.readlines()  

        # 处理数据  
        self.data = []  
        for index in range(len(doc_lines)):  
            # 使用正则表达式移除索引  
            doc_line = re.sub(r'^\d+:\s*', '', doc_lines[index]).strip()  
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

# 自定义 Collate 函数  
def custom_collate_fn(batch):  
    system_content = """You are a code summarizer. Generate ONE-LINE summaries(<=25 words) with this format:  
    genSummary:\tsummary  

    Rules:  
    - Omit explanations  
    - Never repeat doc content  
    """  

    messages = []  
    indices = []  
    for item in batch:  
        prompt = f"""CODE: {item['code']}  
        DOC: {item['doc']}"""  

        message = [  
            {"role": "system", "content": system_content},  
            {"role": "user", "content": prompt}  
        ]  
        messages.append(message)  
        indices.append(item['index'])  

    return messages, indices  

def main():  
    # 预热GPU（避免首次推理延迟）  
    if torch.cuda.is_available():  
        torch.zeros(1).cuda()  

    # 加载模型和分词器  
    model_name = "model/Qwen2.5-Coder-14B-Instruct"  
    start = time.perf_counter()  
    model = AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=torch.float16,  
        device_map="auto",  
        attn_implementation="flash_attention_2"  
    )  
    end = time.perf_counter()  
    print("model生成" + f"执行耗时: {end - start:.6f} 秒")  

    start_t = time.perf_counter()  
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    end_t = time.perf_counter()  
    print("token生成" + f"执行耗时: {end_t - start_t:.6f} 秒")  
    print(torch.cuda.is_available())  

    # 创建数据集和数据加载器  
    dataset = CodeSummaryDataset(  
        'dataset/julia/doc.txt',  
        'dataset/julia/code.txt'  
    )  

    # 配置数据加载器  
    batch_size = 8  # 根据GPU显存调整  
    dataloader = DataLoader(  
        dataset,  
        batch_size=batch_size,  
        shuffle=False,  # 不需要打乱顺序  
        collate_fn=custom_collate_fn,  
        num_workers=os.cpu_count(),  # 根据CPU核心数调整  
        pin_memory=True  # 加速数据传输到GPU  
    )  

    # 结果文件  
    with open('experiment/ds-coder-1_3B/julia_result/julia_ref.txt', 'w', encoding='utf-8') as ref_file:  
        all_results = []  # 存储所有结果  

        # 使用进度条遍历数据加载器  
        for batch_messages, batch_indices in tqdm(dataloader, desc="生成摘要进度"):  
            # 批量处理消息模板  
            batch_texts = [  
                tokenizer.apply_chat_template(  
                    messages,  
                    tokenize=False,  
                    add_generation_prompt=True  
                ) for messages in batch_messages  
            ]  

            # 批量分词  
            model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)  

            # 批量生成  
            generated_ids = model.generate(  
                **model_inputs,  
                max_new_tokens=50,  
                temperature=0.1,  
                repetition_penalty=1.1,  
                num_beams=2,  
                do_sample=False,  
                pad_token_id=tokenizer.eos_token_id  
            )  

            # 解码批量生成结果  
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)  

            # 移除 "genSummary: " 前缀  
            responses = [re.sub(r'^genSummary:\s*', '', r).strip() for r in responses]  

            # 收集结果  
            for local_index, response in zip(batch_indices, responses):  
                all_results.append(f"{local_index}: {response}\n")  
                print(f"第{local_index}生成: {response}")  

        # 一次性写入所有结果  
        ref_file.writelines(all_results)  

    print("所有摘要生成完毕，并已写入文件。")  

if __name__ == '__main__':  
    main()