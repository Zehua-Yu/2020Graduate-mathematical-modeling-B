import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib as mpl
import pandas as pd

results_vec = [[]]
re_tem1 = []
re_tem2 = []
re_tem3 = []
# for i in range(10):
#     df_here = pd.read_csv(f'BP2_results_{i+3}.csv', header=None)
#     for j in range(len(df_here.iloc[:, 0])):
#         re_tem1.append(df_here.iloc[j, 0])
#         re_tem2.append(df_here.iloc[j, 1])
#         re_tem3.append(df_here.iloc[j, 2])

num1 = ['0.4349', '0.5349']
r1 = range(0, 2, 1)
num2 = ['1.871445', '21.871445', '41.871445', '61.871445']
r2 = range(0, 65, 20)
num3 = ['0.79', '10.79', '20.79', '30.79', '40.79', '50.7927']
r3 = range(0, 11, 2)
num4 = ['116.76', '126.76', '136.76']
r4 = range(0, 3, 1)
num5 = ['20.13', '22.13', '24.13', '26.13', '28.13']
r5 = range(0, 5, 1)
num6 = ['130.06', '132.06', '134.06', '136.06', '138.06']
r6 = range(0, 10, 2)
num7 = ['2.7', '22.7', '42.7', '62.7', '82.7', '99.7']
r7 = range(0, 120, 20)
num8 = ['0.23', '0.73', '1.13', '1.73']
r8 = range(0, 16, 4)
num9 = ['5563.2', '6063.2', '6563.2', '7063.2', '8963.2']
r9 = range(0, 35, 7)
num10 = ['2.51', '3.01', '3.51']
r10 = range(0, 15, 5)

num_pa = num10
n = 10
r = r10

df_here = pd.read_csv(f'BP3_results_{n}.csv', header=None)
num = []
for j in range(len(df_here.iloc[:, 0])):
    re_tem1.append(df_here.iloc[j, 0])
    re_tem2.append(df_here.iloc[j, 1])
    re_tem3.append(df_here.iloc[j, 2]*100)
    num.append(f'{j+1}')
fig = plt.figure(figsize=(4, 5))
plt.rcParams['figure.dpi'] = 100000
    # plt.plot(re_tem1)
    # plt.plot(re_tem2)

plt.subplot(311)
ax = plt.gca()
ax.plot(re_tem1, marker="^", markersize=4, lw=1, color='coral')
# ax.plot(re_tem1, lw=1, color='coral')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.axis('tight')
plt.ylabel('S,μg/g', font2)
plt.ylim(2, 5)
plt.tick_params(labelsize=10)
plt.xticks(r, num_pa)
plt.tight_layout()

plt.subplot(312)
ax = plt.gca()
ax.plot(re_tem2, marker="o", markersize=4, lw=1, color='deeppink')
# ax.plot(re_tem2, lw=1, color='deeppink')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.axis('tight')
plt.ylabel('RON', font2)
plt.tick_params(labelsize=10)
plt.xticks(r, num_pa)
plt.tight_layout()

plt.subplot(313)
ax = plt.gca()
ax.plot(re_tem3, marker="*", markersize=4, lw=1, color='deepskyblue')
# ax.plot(re_tem3, lw=1, color='deepskyblue')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.axis('tight')
plt.ylabel('Loss reduction,%', font2)
plt.tick_params(labelsize=10)
plt.xticks(r, num_pa)
plt.tight_layout()

plt.show()