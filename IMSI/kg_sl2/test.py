import pandas as pd

# 读取文本文件
data = pd.read_csv('../1.txt', delimiter='\t')  # 根据实际情况指定分隔符

# 指定列的某几个值
specified_values = ['6', '10', '14']

# 根据指定列的某几个值筛选数据
filtered_data = data[data['指定列名'].isin(specified_values)]

# 打印筛选后的数据
print(filtered_data)

