input_path = '../lua_result/lua_2_NL.txt'
output_path = '../lua_result/lua_2_NL_4819.txt'


def fix_multiline_data(data_lines):
    fixed_lines = []
    buffer = ""

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        if line.split(':', 1)[0].strip()=='4820':break
        # 检查是否是新的索引行（格式为"数字:内容"）
        if ':' in line and line.split(':', 1)[0].strip().isdigit():
            if buffer:  # 如果buffer有内容，先保存
                fixed_lines.append(buffer)
                buffer = ""
            buffer = line
        else:
            # 追加到当前buffer（加空格分隔）
            buffer += " " + line if buffer else line

    if buffer:  # 添加最后一条
        fixed_lines.append(buffer)

    return fixed_lines


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        input_lines = f.readlines()

    fixed_lines = fix_multiline_data(input_lines)
    flag=1
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
        flag+=1

if __name__ == '__main__':
    process_file(input_path, output_path)

# def remove_empty_duplicate_lines(input_file_path, output_file_path):
#     try:
#         with open(input_file_path, 'r', encoding='utf-8') as infile:
#             lines = infile.readlines()
#
#         non_empty_lines = [line for line in lines if line.strip()]
#         unique_lines = []
#         for line in non_empty_lines:
#             if line not in unique_lines:
#                 unique_lines.append(line)
#
#         with open(output_file_path, 'w', encoding='utf-8') as outfile:
#             outfile.writelines(unique_lines)
#         print(f"已成功删除空行和重复行，结果保存于 {output_file_path}")
#     except FileNotFoundError:
#         print("错误：未找到输入文件！")
#     except Exception as e:
#         print(f"错误：出现未知错误：{e}")
#
#
# if __name__ == "__main__":
#     remove_empty_duplicate_lines(output_path, output_path)

#一行出现多个答案处理
# flag = 1
# shiji =1
# last = False
# last_content = ""
# result = []
# with open(input_path, 'r',encoding='UTF-8') as f_in:
#     with open(output_path, 'w',encoding='UTF-8') as f_out:
#         line = f_in.readline()
#         while line:
#
#             if line.find(str(flag))>-1:
#                 if line.find(str(flag+1))>-1 and flag>10 and line.find(str(flag+2))>-1:
#                     f_content = line.split(str(flag+1))[0]
#                     f_out.write(f_content)
#                     result.append(f_content)
#                     temp = str(flag+1)+line.split(str(flag+1))[1].split(str(flag+2))[0]
#                     f_out.write (temp)
#                     result.append(temp)
#                     result.append(line.split(str(flag+2))[1])
#                     last_content = line.split(str(flag+2))[1]
#                     shiji+=3
#                     flag +=3
#                     line = f_in.readline()
#                 if line.find(str(flag+1))>-1 and flag>10:
#                     f_content = line.split(str(flag+1))[0]
#                     f_out.write(f_content)
#                     result.append(f_content)
#
#                     temp = str(flag+1)+line.split(str(flag+1))[1]
#                     f_out.write (temp)
#                     result.append(temp)
#                     last_content = temp
#                     shiji+=2
#                     flag +=2
#                     line = f_in.readline()
#
#                 else:
#                     f_out.write(line)
#                     result.append(line)
#                     last_content = line
#                     flag = flag + 1
#                     shiji+=1
#                     line = f_in.readline()
#             else:
#                 print(flag)
#                 break
# print(len(result))
# written_lines = 0
# with open(output_path, 'w', encoding='UTF-8') as f_out:
#     for i, line in enumerate(result):
#         try:
#             # 确保每行结尾有换行符
#             if not line.endswith('\n'):
#                 line = line + '\n'
#             f_out.write(line)
#             written_lines += 1
#         except Exception as e:
#             print(f"Error writing line {i}: {e}")
#
# print(f"Number of empty lines in result: 0")
# print(f"Number of lines actually written: {written_lines}")
