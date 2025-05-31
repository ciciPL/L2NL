import json

# 输入和输出文件路径
input_file_path = 'Clean_PCSD/train/fixed_20872_36977_3.6v6.4.txt'  # 替换为你的输入文件路径
output_file_path = 'alpacaPCSD/train_fixed_3.6v6.4_without_sentence.json'  # 替换为你想要的输出文件路径

# 系统提示词（适合摘要任务）
system_prompt = "You are an expert code summarization AI. Your task is to generate a concise, ONE-LINE summary based on the provided CODE and its important sentences."
system_prompt_without_sentence = "You are an expert code summarization AI. Your task is to generate a concise, ONE-LINE summary based on the provided CODE."
Instructions = """
1. The summary MUST be a single line.
2. The summary MUST accurately reflect the core functionality described.
3. Do NOT include any explanations or conversational phrases.
4. The provided code and its important sentences will follow this format:<code>USER_INPUT_CODE</code>, <sentences>USER_INPUT_CODE_IMPORTANT_SENTENCES</sentences>
5. Your response MUST strictly follow this format: <summary>YOUR_GENERATED_SUMMARY_HERE</summary>
"""
Instructions_without_sentence = """
1. The summary MUST be a single line.
2. The summary MUST accurately reflect the core functionality described.
3. Do NOT include any explanations or conversational phrases.
4. The provided code and its important sentences will follow this format:<code>USER_INPUT_CODE</code>
5. Your response MUST strictly follow this format: <summary>YOUR_GENERATED_SUMMARY_HERE</summary>
"""
# 创建一个列表来存储转换后的数据
alpaca_data = []

# 打开输入文件并读取内容
with open('../../experiment/EASC/output/python/predictions/python_train_preds.jsonl', 'r',
          encoding='utf-8') as in_preds, \
        open('Clean_PCSD/train/train.jsonl', 'r', encoding='utf-8') as in_code, \
        open("Clean_PCSD/train/fixed_20872_36977_3.6v6.4.txt", 'r', encoding='utf-8') as in_nl:
    codes = []
    for line in in_code:
        code_json = json.loads(line)
        code = code_json.get('raw_code', '')
        codes.append(code.strip())
    print(len(codes))
    nls = []
    for line in in_nl:
        nl = line.split(':')[1]
        nls.append(nl.strip())

    for index, line in enumerate(in_preds):
        # 解析 JSONL 行
        data = json.loads(line)

        # 提取 instruction 和 output
        # instruction = data.get('instruction', '')
        # output = data.get('output', '')

        cleaned_seqs_pred = data.get('cleaned_seqs_pred', '')
        # --- 更新后的重要句子格式化逻辑 ---
        important_sentences_parts = []

        # 计算最大句子编号所占的字符宽度
        # 例如：如果有 9 个句子，最大编号是 "9"，宽度是 1。如果有 12 个句子，最大编号是 "12"，宽度是 2。
        max_digits_in_num = len(str(len(cleaned_seqs_pred)))

        for i, sentence_text in enumerate(cleaned_seqs_pred):
            current_num = i + 1  # 当前句子编号，从1开始

            # 将当前编号转换为字符串，并向右补足空格，使其达到固定宽度 (max_digits_in_num)
            # 例如：如果 max_digits_in_num 是 2:
            #   current_num = 1  -> formatted_num_part = "1 " (数字1后有一个空格)
            #   current_num = 10 -> formatted_num_part = "10" (数字10，宽度已够)
            formatted_num_part = str(current_num).ljust(max_digits_in_num)

            # 构建行前缀。这里使用你例子中的 "sentence" 和全角冒号 "："。
            # 示例："sentence1 ：" 或 "sentence10："
            line_prefix = f"sentence{formatted_num_part}："

            # 确保 sentence_text 是字符串，并去除其两端的空白字符
            if isinstance(sentence_text, str):
                important_sentences_parts.append(f"{line_prefix}{sentence_text.strip()}")
            # else: 如果列表元素不是字符串，你可以在此添加处理逻辑，例如跳过或记录错误

        # 使用换行符将所有格式化后的行连接起来
        sentences_block_content = "\n".join(important_sentences_parts)
        # --- 更新逻辑结束 ---
        nl = nls[index]
        code = codes[index]
        prompt = f"""
        USER_INPUT_CODE:
        <code>
        {code}
        </code>
        
        USER_INPUT_CODE_IMPORTANT_SENTENCES:
        <sentences>
        {sentences_block_content}
        </sentences>
        """

        prompt_without_sentence = f"""
        USER_INPUT_CODE:
        <code>
        {code}
        </code>
        """
        # 创建 Alpaca 格式的字典
        alpaca_format = {
            "instruction": Instructions_without_sentence,
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
    json.dump(alpaca_data, outfile, ensure_ascii=False, indent=2)

print(f"Conversion completed. Output file saved at: {output_file_path}")
