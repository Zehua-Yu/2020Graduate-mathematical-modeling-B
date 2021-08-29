import pandas as pd
from statsmodels.tsa.arima_model import ARMA, ARIMA
import csv
import os
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


diff = 1
df1 = pd.read_excel(f'附件一：325个样本数据处理后.xlsx', header=None)
output_pa = 'prepro_arma.csv'
sum_c = 0
for i in range(354):
    series = []
    if i >= 0:
        for j in range(325):
            series.append(df1.iloc[j+3, i+16])
        p = 5
        q = 0
        print(p, q)
 #       model = ARIMA(timeseries, order=(p, 1, q)).fit()
        model = ARMA(series, order=(p, q)).fit()
        data_h = model.summary()
        print(data_h)
        B = []
        for k in range(p+q):
            B.append(data_h[k])
        with open(output_pa, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(B)
            csv_file.close()
        print(i + 1, 'is done.')
pd_ron = []
for i in range(325):
    pd_ron.append(df1.iloc[i+3, 10])
model1 = ARMA(pd_ron, order=(p, q)).fit()
pd_ron_fit = model1.summary()
B = []
for k in range(p + q):
    B.append(pd_ron_fit[k])
with open(output_pa, "a", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(B)
    csv_file.close()

output_eucl_ntran = f'prepro_arma_ntran.csv'
output_eucl = f'prepro_arma_t.csv'
df4 = pd.read_csv(output_pa, header=None)
#
# #转置用
df4.values
data = df4.iloc[:, :].values
data = list(map(list, zip(*data)))
data = pd.DataFrame(data)
data.to_csv(output_eucl_ntran, header=0, index=0)
df2 = pd.read_csv(output_eucl_ntran, header=None)
vec = []
A = []
i = 0
j = 0
for i in range(326):
    for j in range(5):
        A.append(complex(df2.iloc[j, i])*10)
    vec.append(1)
    vec[i] = A
    A = []
# # # A是一个向量矩阵：euclidean代表欧式距离
distA = pdist(vec, metric='euclidean')
# # # 将distA数组变成一个矩阵
distB = squareform(distA)
for a in range(326):
    A = []
    A.append(distB[a, 325])
    with open(output_eucl, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(A)
        csv_file.close()


#找距离最小的10个
df2 = pd.read_csv('prepro_arma_t.csv', header=None)

tem3 = 0
tem4 = []
for i in range(10):
    tem2 = 0
    tem1 = df2.iloc[0, 0]
    for j in range(325):
        if df2.iloc[j, 0] < tem1:
            if df2.iloc[j, 0] > tem3:
                tem2 = j+1
                tem1 = df2.iloc[j, 0]
    tem3 = tem1
    tem4.append(tem2)
    print(tem2, tem3)
print(tem4)
# print('---------')
# tem3 = 1111111110
# for i in range(10):
#     tem1 = df2.iloc[0, 0]
#     tem2 = 0
#     for j in range(325):
#         if df2.iloc[j, 0] > tem1:
#             if df2.iloc[j, 0] < tem3:
#                 tem2 = j+1
#                 tem1 = df2.iloc[j, 0]
#     tem3 = tem1
#     print(tem2, tem3)

# pa_ch = tem4
# for j in range(325):
#     pa_vec_he =[]
#     for k in range(7):
#         pa_vec_he.append(df1.iloc[j+3, k+2])
#     for i in range(10):
#         pa_he = pa_ch[i]
#         pa_vec_he.append(df1.iloc[j+3, pa_he+15])
#     with open('pre_result_arma.csv', "a", newline="") as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(pa_vec_he)
#         csv_file.close()
