import json

# 输入和输出文件路径
# input_file_path = '../../dataset/ready_sentences_dataset/Clean_PCSD-ast/train/output6k.json'  # 替换为你的输入文件路径
output_file_path = '../../dataset/finetune/alpacaPCSD/train_6k_without_sentence.json'  # 替换为你想要的输出文件路径

# 系统提示词（适合摘要任务）
# system_prompt = "You are an expert code summarization AI. Your task is to generate a concise, ONE-LINE summary based on the provided CODE and its important Snippets."
system_prompt = """
You are an expert code summarization AI. Generate a SINGLE concise, one-line summary by:
1. Primarily analyzing the provided CODE
2. Augmenting with core context from SIMILAR code snippets
"""
system_prompt_without_sentence = """
You are an expert code summarization AI. Your task is to generate a concise, ONE-LINE summary based on the provided CODE.
"""

# 创建一个列表来存储转换后的数据
alpaca_data = []

R_code=[]
with open("../../dataset/LowData/r/code.txt", 'r', encoding='utf8') as f:
    for line in f:
        code = line.split(":",1)[1].strip()
        R_code.append(code)
        if len(R_code) == 3840: break
# 打开输入文件并读取内容
with open('../../script/EASC/output/python/predictions/python_test_preds.jsonl', 'r',
          encoding='utf-8') as in_preds, \
        open('../../dataset/finetune/aliPCSDData/output6k.json', 'r', encoding='utf-8') as in_code, \
        open("../../dataset/finetune/aliPCSDData/output6k.json", 'r', encoding='utf-8') as in_nl:
    codes = []
    # for line in in_code:
    #     code = line.split(":",1)[1].strip()
    #     codes.append(code)
        # if len(codes) ==3840: break
    for line in in_code:
        code_json = json.loads(line)
        code = code_json.get('raw_code', '')
        codes.append(code.strip())
    # print(len(codes))
    nls = []
    # for line in in_nl:
    #     nl = line.split(':')[1]
    #     nls.append(nl.strip())
    for lines in in_nl:
        nl_json = json.loads(lines)
        nl = nl_json.get('comment', '')
        nls.append(nl.strip())
    faith_nl = 0
    for index, line in enumerate(in_preds):
        if index ==6470:break
        # 解析 JSONL 行
        data = json.loads(line)

        # 提取 instruction 和 output
        # instruction = data.get('instruction', '')
        # output = data.get('output', '')

        # cleaned_seqs_pred = data.get('cleaned_seqs_pred', [R_code[index]])
        #
        # # --- 更新后的重要句子格式化逻辑 ---
        # important_sentences_parts = []
        # max_digits_in_num = len(str(len(cleaned_seqs_pred)))
        # for i, sentence_text in enumerate(cleaned_seqs_pred):
        #     current_num = i + 1  # 当前句子编号，从1开始
        #     formatted_num_part = str(current_num).ljust(max_digits_in_num)
        #     line_prefix = f"Code Snippet{formatted_num_part}："
        #     if isinstance(sentence_text, str):
        #         important_sentences_parts.append(f"{line_prefix}{sentence_text.strip()}")
        # sentences_block_content = "\n".join(important_sentences_parts)
        # --- 更新逻辑结束 ---
        # nl = nls[index]
        # if nl.find('<summary>')>-1:
        #     if nl.strip() == '<summary>':
        #         faith_nl+=1
        #         continue
        #     else:
        #         nl = nl.strip().replace('<summary>', '').replace('</summary>', '')
        code = codes[index]
        nl = nls[index]
        # prompt = f"""
        # ### USER INPUT CODE ###
        # {code}
        #
        # ### CORE CONTEXT SNIPPETS ###
        # {sentences_block_content}
        # """

        prompt_without_sentence = f"""
        ### USER INPUT CODE ###
        {code}
        """
        instruction = """
        Generate a SINGLE concise one-line code summary by:
        1. Primarily analyzing the main code under ### USER INPUT CODE ###
        2. Augmenting with core context under ### CORE CONTEXT SNIPPETS ###
        Focus on core functionality and ignore implementation details.
        """
        instruction_without_sentences = """
        Generate a SINGLE concise one-line code summary by:
        1. Analyzing the main code under ### USER INPUT CODE ###
        Focus on core functionality and ignore implementation details.
        """
        # if len(cleaned_seqs_pred) > 0:
        #     instruction = i
        # else:
        #     instruction = i_without_sentence
        #     prompt = prompt_without_sentence
        # 创建 Alpaca 格式的字典
        i = f"""
        Code: {code}"""
        alpaca_format = {
            "instruction": instruction_without_sentences,
            "input": prompt_without_sentence,  # 用户输入（选填），这里留空
            "output": nl,
            "system": system_prompt_without_sentence,
            "history": []  # 历史记录留空
        }

        # 将 Alpaca 格式的字典添加到列表中
        alpaca_data.append(alpaca_format)

    # 将列表写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    print(len(alpaca_data))
    print(faith_nl)
    json.dump(alpaca_data, outfile, ensure_ascii=False, indent=2)

print(f"Conversion completed. Output file saved at: {output_file_path}")
