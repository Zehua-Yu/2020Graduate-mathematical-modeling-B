import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df1 = pd.read_excel('附件一：325个样本数据处理后.xlsx', header=None)
df2 = pd.read_excel('附件三：285号和313号样本原始数据操作变量.xlsx', header=None)
df4 = pd.read_excel('附件四：354个操作变量信息.xlsx', header=None)

# RF method
data_fea = df1.iloc[3:328, 16:370]

model = RandomForestRegressor(random_state=1, max_depth=10)
data_fea = data_fea.fillna(0)
data_fea = pd.get_dummies(data_fea)
model.fit(data_fea, df1.iloc[3:328, 11])
features = data_fea.columns
importances = model.feature_importances_
print(features)
indices = np.argsort(importances[0:354])
plt.title('Index selection')
plt.barh(range(len(indices)), importances[indices], color='pink', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative importance of indicators')
plt.show()

# fangcha
# tem2 = []
# var = []
# for i in range(354):
#     var.append(np.var(df1.iloc[3:328, 16+i]))
# re = []
# tem1 = 0
# num1 = 0
# for i in range(30):
#     if i == 0:
#         for j in range(len(var)):
#             if var[j] >= tem1:
#                 tem1 = var[j]
#                 num1 = j
#         re.append(tem1)
#         tem2 = tem1
#         tem1 = 0
#         print(num1)
#     else:
#         for j in range(len(var)):
#             if var[j] >= tem1:
#                 if var[j] < tem2:
#                     tem1 = var[j]
#                     num1 = j
#         re.append(tem1)
#         tem2 = tem1
#         tem1 = 0
#         print(num1)
# print(re)