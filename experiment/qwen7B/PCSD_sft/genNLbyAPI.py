import dashscope
import pandas as pd
import time


def process_file(input_file):
    # 读取Excel文件，假设文件名为 'data.xlsx'
    file_path = input_file

    # 使用 pandas 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 将两列数据分别转为 list
    prompt_list = df['Prompt'].tolist()
    completion_list = df['Completion'].tolist()
    return prompt_list, completion_list


def process_txt_file(input_file):
    # 读取txt文件,取4051个
    prompt_list = []
    with open(input_file) as f:
        for line in f:
            index = line.split(':')[0]
            if index == '4052': break
            prompt_list.append(line.replace(index+':\t', ''))

    return prompt_list


def generate_summary(prompt_list, api_key):
    results = []

    for idx, prompt in enumerate(prompt_list):
        print(f"Processing prompt {idx + 1}/{len(prompt_list)}: {prompt[:30]}...")
        prompt_new = f"""Generate a ONE-LINE summary (≤30 words) for this code:\nCode: {prompt}\nSummary:"""
        try:
            response = dashscope.Generation.call(
                api_key=api_key,
                model="qwen2.5-7b-instruct-ft-202505042236-e5cd",
                prompt=prompt,
                result_format='message'
            )

            if response.output is None or not hasattr(response.output, 'choices'):
                print(f"⚠️ No valid output returned for prompt: {prompt}")
                print("Full error response:", response)
                results.append(None)
                time.sleep(1)  # 如果失败，多等一会儿再试
                continue

            result = response.output.choices[0].message.content
            print(result+'\n')
            results.append(result)

        except Exception as e:
            print(f"❌ Error occurred with prompt: {prompt}")
            print("Exception:", str(e))
            results.append(None)

        time.sleep(0.1)  # 控制请求频率
    return results


def save_refs(refs_list):
    with open('../dataset/aliPCSDData/test_500_refs.txt', 'w', encoding='utf-8') as f:
        for idx, content in enumerate(refs_list, start=1):
            f.write(f"{idx}:{content}\n")


def save_hyps(hyps_list):
    with open('rkt_python_nl_7B_sft_4051.txt', 'w', encoding='utf-8') as f:
        for idx, content in enumerate(hyps_list, start=1):
            f.write(f"{idx}:{content}\n")


if __name__ == '__main__':
    api_key = 'sk-5f1dde7966184692b7083e2bd7f9fa3f'
    input_file = '../../../experiment/ds-coder-1_3B/rkt_result/rkt_2_python_40510.txt'
    prompts= process_txt_file(input_file)

    # save_refs(refs)
    hyps = generate_summary(prompts, api_key)
    save_hyps(hyps)
