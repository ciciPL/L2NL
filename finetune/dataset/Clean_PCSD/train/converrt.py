def add_index_to_file(input_file, output_file):
    """
    读取文件，为每一行数据添加索引，并将结果保存到新的文件中。

    Args:
        input_file (str): 输入文件路径。
        output_file (str): 输出文件路径。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8') as outfile:

            for index, line in enumerate(infile, start=1):
                # 去除行首尾的空白字符
                line = line.strip()
                # 添加索引并写入输出文件
                outfile.write(f"{index}: {line}\n")

        print(f"✅ Successfully added index to {input_file} and saved to {output_file}")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"❌ Error: An error occurred: {e}")


if __name__ == "__main__":
    input_file = "code"  # 替换为您的输入文件路径
    output_file = "code_index.txt"  # 替换为您的输出文件路径

    add_index_to_file(input_file, output_file)