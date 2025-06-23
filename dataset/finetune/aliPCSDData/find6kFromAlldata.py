import json


# 读取 aliref.txt 获取所有摘要
def read_abstracts(file_path):
    abstracts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':', 1)
                abstract = parts[1].strip()
                if abstract:
                    abstracts.append(abstract)
    print(len(abstracts))
    return abstracts


# 匹配 train.json 中 comment 是否包含某个摘要
def match_comments(jsonl_file, abstracts, output_file):
    matched_count = 0

    with open(jsonl_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:
        for ab in abstracts:
            for line in fin:
                try:
                    item = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue  # 跳过无效行

                comment = item.get('comment', '')
                if ab in comment:
                    fout.write(line)  # 写入整行
                    matched_count += 1
                    break  # 只要有一个摘要匹配就保留这条数据

    print(f"共找到 {matched_count} 条匹配数据，已保存至 {output_file}")


# 主程序入口
if __name__ == "__main__":
    aliref_file = "aliref.txt"
    train_file = "../Clean_PCSD/train/train.jsonl"
    output_file = "output6k.json"

    abstracts = read_abstracts(aliref_file)
    match_comments(train_file, abstracts, output_file)
