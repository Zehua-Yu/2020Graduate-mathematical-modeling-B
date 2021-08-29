import pandas as pd
import numpy as np
import csv
import xlrd
from xlutils.copy import copy

df1 = pd.read_excel('附件一：325个样本数据.xlsx', header=None)
df2 = pd.read_excel('附件三：285号和313号样本原始数据操作变量.xlsx', header=None)
df4 = pd.read_excel('附件四：354个操作变量信息.xlsx', header=None)

#判断两个数据集是否一致
#285
# for i in range(354):
#     pa = []
#     for j in range(40):
#         pa.append(df2.iloc[3+j, 1+i])
#     m_pa = round(np.mean(pa), 5)
#     df1pa = round(df1.iloc[287, 16+i], 5)
#     if df1pa != m_pa:
#         print(m_pa, df1pa)
#
# #313
# for i in range(354):
#     pa = []
#     for j in range(40):
#         pa.append(df2.iloc[44+j, 1+i])
#     m_pa = round(np.mean(pa))
#     df1pa = round(df1.iloc[315, 16+i])
#     if df1pa != m_pa:
#         print(m_pa, df1pa)


#提取数据

for i in range(200):
    coll = []
    coll.append(df1.iloc[i+3, 9])
    coll.append(df1.iloc[i+3, 11])
    for j in range(28):
        coll.append(df1.iloc[i+3, 16+j])
    with open('pre_re.csv', "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(coll)
        csv_file.close()
    print(i)