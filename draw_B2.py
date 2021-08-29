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

num1 = ['1', '2', '3', '4']
num2 = ['1', '60', '120', '180', '240']
num3 = ['2', '5', '8', '11', '14']
num4 = ['1', '2', '3', '4', '5', '6']
num5 = ['2', '5', '8', '11', '14']
num6 = ['1', '2', '3', '4']
num7 = ['1', '20', '40', '60', '80', '100']
num8 = ['1', '10', '20', '30', '40', '50']
num9 = ['2', '4', '6', '8', '10', '12', '14']
num10 = ['1', '5055', '10110', '15165', '20220']

num_pa = num10
n = 10
r = range(0, 20220+5055, 5055)

df_here = pd.read_csv(f'BP2_results_{n}.csv', header=None)
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
# ax.plot(re_tem1, marker="^", markersize=4, lw=1, color='coral')
ax.plot(re_tem1, lw=1, color='coral')
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
# ax.plot(re_tem2, marker="o", markersize=4, lw=1, color='deeppink')
ax.plot(re_tem2, lw=1, color='deeppink')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.axis('tight')
plt.ylabel('RON', font2)
plt.tick_params(labelsize=10)
plt.xticks(r, num_pa)
plt.tight_layout()

plt.subplot(313)
ax = plt.gca()
# ax.plot(re_tem3, marker="*", markersize=4, lw=1, color='deepskyblue')
ax.plot(re_tem3, lw=1, color='deepskyblue')
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
plt.axis('tight')
plt.ylabel('Loss reduction,%', font2)
plt.tick_params(labelsize=10)
plt.xticks(r, num_pa)
plt.tight_layout()

plt.show()