
# 读取文本文件
with open('test1.txt', 'r') as file:
    lines = file.readlines()

# 处理数据，将第二列的值设置为1
modified_lines = []
for line in lines:
    if line.strip():  # 忽略空行
        data = line.strip().split()  # 分割每行数据
        data[1] = '1'  # 将第二列的值设置为1
        modified_lines.append(' '.join(data) + '\n')
    else:
        modified_lines.append('\n')  # 保留空行

# 将修改后的数据写回文本文件
with open('test2.txt', 'w') as file:
    file.writelines(modified_lines)

