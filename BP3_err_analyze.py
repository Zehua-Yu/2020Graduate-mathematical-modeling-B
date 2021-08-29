import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import csv
def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    print(v)
    print(v_)
    return np.mean(np.abs(v_ - v))

def Adjustment_pa(start, end, step, base):
    if base - start < step:
        start = base
    else:
        start = base - (int((base - start) / step) * step)
    num_of_cal = int((end - start) / step)
    pa_vec = []
    pa_vec.append(start)
    for i in range(num_of_cal):
        pa_vec.append(round(pa_vec[i] + step, 2))
    return pa_vec


def Adjustment_pa1(start, times, step):
    num_of_cal = 2 * times + 1
    ini_st = start - times * step
    pa_vec = []
    pa_vec.append(ini_st)
    for i in range(num_of_cal - 1):
        pa_vec.append(round(pa_vec[i] + step, 2))
    return pa_vec


a_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8, 0.5349280025, 187.1445, 31.9275328675, 136.76122, 26.134938,
         134.05572, 4.264814875, 0.438199295, 8663.223025, 2.91078745]

pa_316 = Adjustment_pa(0.4, 0.6, 0.1, a_133[7])
pa_54 = Adjustment_pa(1.5, 650, 10, a_133[8])  # -125
pa_43 = Adjustment_pa(-1, 54, 5, a_133[9])
pa_21 = Adjustment_pa(110, 140, 10, a_133[10])
pa_236 = Adjustment_pa(20, 30, 2, a_133[11])
pa_171 = Adjustment_pa(130, 140, 1, a_133[12])
pa_259 = Adjustment_pa(2, 100, 1, a_133[13])
pa_324 = Adjustment_pa(0.2, 1.8, 0.1, a_133[14])
pa_53 = Adjustment_pa(5500, 9000, 100, a_133[15])
pa_214 = Adjustment_pa(2.5, 4.0, 0.1, a_133[16])  # 250

tr_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]

pa_vec = [pa_316, pa_54, pa_43, pa_21, pa_236, pa_171, pa_259, pa_324, pa_53, pa_214]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
df1 = pd.read_csv('pre_result_arma.csv', header=None)
df2 = pd.read_excel('附件一：325个样本数据处理后.xlsx', header=None)
df_gt = pd.read_excel('S-RON.xlsx', header=None)
x_tem1 = []
y_tem1 = []
y = np.zeros((324, 2))
x = np.zeros((17, 324))
for j in range(17):
    for a in range(325):
        if a != 132:
            x_tem1.append(df1.iloc[a, j])
    x[j] = x_tem1
    x_tem1 = []

for i in range(324):
    for k in range(2):
        y_tem1.append(df_gt.iloc[i, k])
    y[i] = y_tem1
    y_tem1 = []
# for i in range(325):
#     for k in range(2):
#         y_tem1.append(df2.iloc[i + 3, k + 9])
#     y[i] = y_tem1
#     y_tem1 = []
#     x[i] = x_tem1


x_t = np.array(x, dtype='float32').T  # [148]
y_true = np.array(y, dtype='float32')  # [141]

mm = MinMaxScaler()  # 实例化
std = mm.fit(x_t)  # 训练模型
x_true = std.transform(x_t)  # 转化

X = tf.placeholder(tf.float32, [None, 17])
Y = tf.placeholder(tf.float32, [None, 2])

w1 = tf.Variable(tf.truncated_normal([17, 9], stddev=0.1))
b1 = tf.Variable(tf.zeros([9]))

w2 = tf.Variable(tf.zeros([9, 2]))
b2 = tf.Variable(tf.zeros([2]))
L1 = tf.nn.relu(tf.matmul(X, w1) + b1)
y_pre = tf.matmul(L1, w2) + b2
loss = tf.reduce_mean(tf.cast(tf.square(Y - y_pre), tf.float32))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, './BP3/BP_model')
    gt_heres = []
    pre_s = []
    gt_herer = []
    pre_r = []
    for i in range(325):
        test_sa = []
        for j in range(17):
            test_sa.append(df1.iloc[i, j])
        gt_heres.append(df2.iloc[i + 3, 9])
        gt_herer.append(df2.iloc[i + 3, 10])
        x_test = np.array([test_sa], dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pre_s.append(result[0][0])
        pre_r.append(result[0][1])
    gt_herer = np.array(gt_herer)
    gt_heres = np.array(gt_heres)
    pre_r = np.array(pre_r)
    pre_s = np.array(pre_s)
    print(MAE(gt_heres, pre_s), MAE(gt_herer, pre_r))
    print(RMSE(gt_heres, pre_s), RMSE(gt_herer, pre_r))
    print(MAPE(gt_heres, pre_s), MAPE(gt_herer, pre_r))