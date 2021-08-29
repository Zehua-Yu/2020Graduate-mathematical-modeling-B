import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import csv


def Adjustment_pa(start, end, step, base):
    if base - start < step:
        start = base
    else:
        start = base - (int((base - start)/step)*step)
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

a_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8, 0.03863123350000001, 0.35064612999999994, 8.84629, -0.15295047525,
         34.452483750000006, 0.5315088624999998, 120.767935, 124.65896499999997, 0.6106065250000001, 212.80990000000014]


heat_vas_in_64 = Adjustment_pa(0, 0.2, 0.05, a_133[7])
D101_354 = Adjustment_pa(-125, 0.5, 0.5, a_133[8])#-125
P101B_246 = Adjustment_pa(-1.5, 12, 1, a_133[9])
S_22 = Adjustment_pa(0, 5, 1, a_133[10])
S_20 = Adjustment_pa(30, 45, 1, a_133[11])
wind_40 = Adjustment_pa(0.55, 0.7, 0.05, a_133[12])
E101D_120 = Adjustment_pa(50, 150, 1, a_133[13])
stable_t_16 = Adjustment_pa(100, 150, 1, a_133[14])
D125_131 = Adjustment_pa(-0.85, 2, 0.2, a_133[15])
K103A_265 = Adjustment_pa(-20000, 220, 1, a_133[16])#250



tr_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]

pa_vec = [heat_vas_in_64, D101_354, P101B_246, S_22, S_20, wind_40, E101D_120, stable_t_16, D125_131, K103A_265]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
df1 = pd.read_csv('rf_features10.csv', header=None)
df2 = pd.read_excel('附件一：325个样本数据处理后.xlsx', header=None)
x_tem1 = []
y_tem1 = []
y = np.zeros((300, 2))
x = np.zeros((17, 300))
for j in range(17):
    for a in range(300):
        x_tem1.append(df1.iloc[a, j])
    x[j] = x_tem1
    x_tem1 = []
for i in range(300):
    for k in range(2):
        y_tem1.append(df2.iloc[i + 3, k + 9])
    y[i] = y_tem1
    y_tem1 = []
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
    # for i in range(1, 100):
    #     for j in range(len(y_true)):
    #         sess.run(train_op, feed_dict={X: [x_true[j, :]], Y: [y_true[j, :]]})
    #         print('第 % s批次第 % s个样本训练的损失为: % s, 真实值为： % s, 预测值为: % s' % (i, j + 1,
    #                                                                     sess.run(loss, feed_dict={X: [x_true[j, :]],
    #                                                                                               Y: [y_true[j, :]]}),
    #                                                                     y_true[j, :],
    #                                                                     sess.run(y_pre, feed_dict={X: [x_true[j, :]],
    #                                                                                                Y: [y_true[j, :]]})))
    #         if j%10 == 0:
    #             saver.save(sess,'./BP2/BP_model')
    saver.restore(sess, './BP2/BP_model')
    x_test = np.array([a_133], dtype='float32')
    x_test = std.transform(x_test)
    result = sess.run(y_pre, feed_dict={X: x_test})
    RON_O = 89.4 + ((result[0][1] - 88.09)/88.09)*89.4
    S_M = 5
    tem1 = result[0][1]
    print(RON_O, tem1)

    # pa1

    pa_tem1 = pa_vec[0][0]
    pr_vec = []
    for i in range(len(pa_vec[0])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_vec[0][i])
        for j in range(9):
            tr_133[0].append(a_133[j + 8])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_1.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem1 = pa_vec[0][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('1')
    print(result[0][1], RON_O)
    print(tr_133, np.abs(1.33 - (RON_O - result[0][1]))/1.33)

    pa_tem2 = pa_vec[1][0]
    for i in range(len(pa_vec[1])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_vec[1][i])
        for j in range(8):
            tr_133[0].append(a_133[j + 9])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append((1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_2.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem2 = pa_vec[1][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('2')

    pa_tem3 = pa_vec[2][0]
    for i in range(len(pa_vec[2])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_vec[2][i])
        for j in range(7):
            tr_133[0].append(a_133[j + 10])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_3.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem3 = pa_vec[2][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('3')

    pa_tem4 = pa_vec[3][0]
    for i in range(len(pa_vec[3])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_vec[3][i])
        for j in range(6):
            tr_133[0].append(a_133[j + 11])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_4.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem4 = pa_vec[3][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('4')

    pa_tem5 = pa_vec[4][0]
    for i in range(len(pa_vec[4])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_vec[4][i])
        for j in range(5):
            tr_133[0].append(a_133[j + 12])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_5.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem5 = pa_vec[4][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('5')

    pa_tem6 = pa_vec[5][0]
    for i in range(len(pa_vec[5])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_vec[5][i])
        for j in range(4):
            tr_133[0].append(a_133[j + 13])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_6.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem6 = pa_vec[5][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('6')

    pa_tem7 = pa_vec[6][0]
    for i in range(len(pa_vec[6])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_tem6)
        tr_133[0].append(pa_vec[6][i])
        for j in range(3):
            tr_133[0].append(a_133[j + 14])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_7.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem7 = pa_vec[6][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('7')

    pa_tem8 = pa_vec[7][0]
    for i in range(len(pa_vec[7])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_tem6)
        tr_133[0].append(pa_tem7)
        tr_133[0].append(pa_vec[7][i])
        for j in range(2):
            tr_133[0].append(a_133[j + 15])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_8.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem8 = pa_vec[7][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('8')

    pa_tem9 = pa_vec[8][0]
    for i in range(len(pa_vec[8])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_tem6)
        tr_133[0].append(pa_tem7)
        tr_133[0].append(pa_tem8)
        tr_133[0].append(pa_vec[8][i])
        for j in range(1):
            tr_133[0].append(a_133[j + 15])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_9.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem9 = pa_vec[8][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('9')

    pa_tem10 = pa_vec[9][0]
    for i in range(len(pa_vec[9])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_tem6)
        tr_133[0].append(pa_tem7)
        tr_133[0].append(pa_tem8)
        tr_133[0].append(pa_tem9)
        tr_133[0].append(pa_vec[9][i])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        pr_vec.append(result[0][0])
        pr_vec.append(result[0][1])
        pr_vec.append(np.abs(1.33 - (RON_O - result[0][1])) / 1.33)
        with open('BP2_results_10.csv', "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(pr_vec)
            csv_file.close()
        pr_vec = []
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem10 = pa_vec[9][i]
                    print(result[0][0], result[0][1], np.abs(1.33 - (RON_O - result[0][1]))/1.33)
    print('10')
    print(tr_133)
    print(result[0][1], RON_O, np.abs(1.33 - (RON_O - result[0][1]))/1.33)