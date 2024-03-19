





# import pandas as pd
#
# # 读取txt文件，假设没有列名
# df = pd.read_csv('1.txt', delimiter='\t', header=None)
#
# # 选择需要的列和条件，假设要选择第1列和第3列
# selected_data = df.loc[(df[0] == '2') | (df[0] == '10'), 1]
# # selected_data = df.loc[(df['column_name'] == '2') | (df['column_name'] == '10'), ['co', 'column_2']]
# # 输出结果
# print(selected_data)
#
# import csv
#
# # 打开txt文件
# with open('1.txt', 'r') as file:
#     reader = csv.reader(file, delimiter='\t')
#     # 选择需要的列和条件，假设要选择第1列和第3列
#     selected_data = [row for row in reader if row[2] == 1 or row[0] == 5]
#
# # 输出结果
# for row in selected_data:
#     print(row)

# import pandas as pd
#
# # 读取txt文件，假设没有列名
# df = pd.read_csv('1.txt', delimiter='\t', header=None)
#
# # 选择需要的列和条件，假设要选择第1列和第3列
# selected_data = df.loc[(df.iloc[:, 0] == 1) | (df.iloc[:, 0] == 6), [1]]
# # 进行切片操作，选择B，C，D，E四列区域内，B列大于6的值
#
#
# df.loc()
# # 输出结果
# print(selected_data)

# 打开txt文件
with open('valid.txt', 'r') as file:
    with open('valid1.txt','a') as target:
        # 逐行读取文件内容
        lines = file.readlines()

        for line in lines:
            [a,b,c] = line.replace('\n','').split('	')
            if b in ['19', '3', '20','6']:
                target.write(line)
                # print(line)
        # if a=='22620':
        #     break

    # 选择需要的行和条件，假设要选择第1列（索引为0）中的值为value_1或value_2的行
    # selected_lines = [line for line in lines if line.split()[2] in ['t6', 't2']]

# 输出结果
# for line in selected_lines:
#     print(line)












