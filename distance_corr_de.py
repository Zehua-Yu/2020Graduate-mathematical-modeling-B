import pandas as pd
import numpy as np
import csv

from scipy.spatial.distance import pdist, squareform
import numpy as np



def distcorr(X, Y):
    """ Compute the distance correlation function

    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

df1 = pd.read_excel('附件一：325个样本数据.xlsx', header=None)
df2 = pd.read_excel('附件三：285号和313号样本原始数据操作变量.xlsx', header=None)
df4 = pd.read_excel('附件四：354个操作变量信息.xlsx', header=None)

ron_t1 = []
for i in range(325):
    ron_t1.append(round(df1.iloc[i+3, 11], 2))

dis_corr = []
for j in range(354):
    pa_tem1 = []
    for k in range(325):
        pa_tem1.append(df1.iloc[3+k, 16+j])
    dis_corr.append(distcorr(ron_t1, pa_tem1))


tem2 = []
re = []
tem1 = 0
num1 = 0
num_tem = []
for i in range(23):
    if i == 0:
        for j in range(len(dis_corr)):
            if dis_corr[j] >= tem1:
                tem1 = dis_corr[j]
                num1 = j + 1
        re.append(tem1)
        tem2 = tem1
        tem1 = 0
        print(num1)
        num_tem.append(num1)
    else:
        for j in range(len(dis_corr)):
            if dis_corr[j] >= tem1:
                if dis_corr[j] < tem2:
                    tem1 = dis_corr[j]
                    num1 = j + 1
        re.append(tem1)
        tem2 = tem1
        tem1 = 0
        print(num1)
        num_tem.append(num1)
print(re)
print(num_tem)
print(len(num_tem))

re_tem1 = []
for i in range(325):
    re_tem1.append(df1.iloc[i + 3, 2])
    re_tem1.append(df1.iloc[i + 3, 3])
    re_tem1.append(df1.iloc[i + 3, 4])
    re_tem1.append(df1.iloc[i + 3, 5])
    re_tem1.append(df1.iloc[i + 3, 6])
    re_tem1.append(df1.iloc[i + 3, 7])
    re_tem1.append(df1.iloc[i + 3, 8])
    for j in range(23):
        re_tem1.append(df1.iloc[i+3, int(num_tem[j])+16])
    re_tem1.append(df1.iloc[i + 3, 9])
    re_tem1.append(df1.iloc[i + 3, 10])
    with open('pre_re_dis_corr.csv', "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(re_tem1)
        csv_file.close()
    re_tem1 = []