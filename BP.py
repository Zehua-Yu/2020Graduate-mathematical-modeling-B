import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os


def Adjustment_pa(start, end, step):
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


a_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8, 0.6679688000000004, 229.49908500000006, 80.11344675,
         6.879375000000001,
         430.13518750000003, 0.31505031499999986, 6.9461129999999995, -0.0003515624999999998, 25.778817249999996,
         4.423842574999999, 719.6723600000003, 10.773180000000009, 82.57084174999997, 50.51884974999999,
         419.33809249999996, 23.801855000000007, 0.16999999999999996, 51.72783425, 73.29910000000004, 16.27565,
         24.427310000000013, 10, 3.9984560000000022]

prewarm_out_temp_328 = Adjustment_pa(0.6679688, 300, 2)
Mpa_water_out_flow_37 = Adjustment_pa(3, 450, 50)#6500
con_out_temp_329 = Adjustment_pa(0.001, 160, 2)#400
E101DEF_out_temp_338 = Adjustment_pa(-1, 50, 1)#150
E101DEF_pip_out_temp_336 = Adjustment_pa(400, 450, 1)
E101_shell_out_339 = Adjustment_pa(5, 150, 1)
E101ABC_pip_out_temp_334 = Adjustment_pa(0, 380, 2)
E101ABC_out_temp_337 = Adjustment_pa(0, 200, 2)
tt_back_18 = Adjustment_pa(40, 800, 50)
re_sa_341 = Adjustment_pa(15, 45, 1)
PDI2104_189 = Adjustment_pa(15, 45, 1)
re_down_8 = Adjustment_pa(50, 110, 5)
re_weight_342 = Adjustment_pa(2.95, 7, 0.5)
oil1_flow_46 = Adjustment_pa(0, 140, 5)
R101_bed_middle_temp_238 = Adjustment_pa(400, 500, 1)
filter_ME101_out_temp_325 = Adjustment_pa(400, 450, 1)
PIC11_243 = Adjustment_pa(25, 110, 10)
A202A_out_temp_173 = Adjustment_pa(30, 100, 1)
K103A_gas_263 = Adjustment_pa(-2000, 45, 1)
K101_gas_stress_280 = Adjustment_pa(-2, 4000, 1)
K101_ingas_stress_281 = Adjustment_pa(-0.5, 4000, 100)
EH101_out_176 = Adjustment_pa(-165000, 430, 50)
K103A_gas_strees_262 = Adjustment_pa(0.05, 400000, 1000)

tr_133 = [248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]
pa_vec = [prewarm_out_temp_328, Mpa_water_out_flow_37, con_out_temp_329, E101DEF_out_temp_338, E101DEF_pip_out_temp_336,
          E101_shell_out_339, E101ABC_pip_out_temp_334, E101ABC_out_temp_337, tt_back_18, re_sa_341, PDI2104_189,
          re_down_8, re_weight_342, oil1_flow_46, R101_bed_middle_temp_238, filter_ME101_out_temp_325, PIC11_243,
          A202A_out_temp_173, K103A_gas_263, K101_gas_stress_280, K101_ingas_stress_281, EH101_out_176, K103A_gas_strees_262]




os.environ["CUDA_VISIBLE_DEVICES"] = "0"
df1 = pd.read_csv('pre_re_dis_corr.csv', header=None)
df2 = pd.read_excel('附件一：325个样本数据.xlsx', header=None)
x_tem1 = []
y_tem1 = []
y = np.zeros((260, 2))
x = np.zeros((30, 260))
for j in range(30):
    for a in range(260):
        x_tem1.append(df1.iloc[a, j])
    x[j] = x_tem1
    x_tem1 = []
for i in range(260):
    for k in range(2):
        y_tem1.append(df2.iloc[i + 3, k + 9])
    y[i] = y_tem1
    y_tem1 = []
#     x[i] = x_tem1

print(len(x))
print(len(y))
x_t = np.array(x, dtype='float32').T  # [148]
y_true = np.array(y, dtype='float32')  # [141]

mm = MinMaxScaler()  # 实例化
std = mm.fit(x_t)  # 训练模型
x_true = std.transform(x_t)  # 转化

X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 2])

w1 = tf.Variable(tf.truncated_normal([30, 15], stddev=0.1))
b1 = tf.Variable(tf.zeros([15]))

w2 = tf.Variable(tf.zeros([15, 2]))
b2 = tf.Variable(tf.zeros([2]))
L1 = tf.nn.relu(tf.matmul(X, w1) + b1)
y_pre = tf.matmul(L1, w2) + b2
loss = tf.reduce_mean(tf.cast(tf.square(Y - y_pre), tf.float32))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
RON_O = 89.4
S_M = 5
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(1, 50):
        for j in range(len(y_true)):
            sess.run(train_op, feed_dict={X: [x_true[j, :]], Y: [y_true[j, :]]})
            print('第 % s批次第 % s个样本训练的损失为: % s, 真实值为： % s, 预测值为: % s' % (i, j + 1,
                                                                        sess.run(loss, feed_dict={X: [x_true[j, :]],
                                                                                                  Y: [y_true[j, :]]}),
                                                                        y_true[j, :],
                                                                        sess.run(y_pre, feed_dict={X: [x_true[j, :]],
                                                                                                   Y: [y_true[j, :]]})))
            if j%10 == 0:
                saver.save(sess,'./BP_model')
    saver.restore(sess, './BP_model')

    # pa1
    tem1 = 85.963394
    pa_tem1 = pa_vec[0][0]
    for i in range(len(pa_vec[0])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_vec[0][i])
        for j in range(22):
            tr_133[0].append(a_133[j + 8])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem1 = pa_vec[0][i]
                    print(result[0][0], result[0][1])
    print('1')

    pa_tem2 = pa_vec[0][1]
    for i in range(len(pa_vec[1])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_vec[1][i])
        for j in range(21):
            tr_133[0].append(a_133[j + 9])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem2 = pa_vec[1][i]
                    print(result[0][0], result[0][1])
    print('2')

    pa_tem3 = pa_vec[0][2]
    for i in range(len(pa_vec[2])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_vec[2][i])
        for j in range(20):
            tr_133[0].append(a_133[j + 10])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <=RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem3 = pa_vec[2][i]
                    print(result[0][0], result[0][1])
    print('3')

    pa_tem4 = pa_vec[0][3]
    for i in range(len(pa_vec[3])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_vec[3][i])
        for j in range(19):
            tr_133[0].append(a_133[j + 11])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem4 = pa_vec[3][i]
                    print(result[0][0], result[0][1])
    print('4')

    pa_tem5 = pa_vec[0][4]
    for i in range(len(pa_vec[4])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_vec[4][i])
        for j in range(18):
            tr_133[0].append(a_133[j + 12])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem5 = pa_vec[4][i]
                    print(result[0][0], result[0][1])
    print('5')

    pa_tem6 = pa_vec[0][5]
    for i in range(len(pa_vec[5])):
        tr_133 = [[248, 89.4, 55.9, 20.6, 23.5, 50.11, 727.8]]
        tr_133[0].append(pa_tem1)
        tr_133[0].append(pa_tem2)
        tr_133[0].append(pa_tem3)
        tr_133[0].append(pa_tem4)
        tr_133[0].append(pa_tem5)
        tr_133[0].append(pa_vec[5][i])
        for j in range(17):
            tr_133[0].append(a_133[j + 12])
        x_test = np.array(tr_133, dtype='float32')
        x_test = std.transform(x_test)
        result = sess.run(y_pre, feed_dict={X: x_test})
        if result[0][0] <= 5:
            if result[0][1] <= RON_O:
                if result[0][1] >= tem1:
                    tem1 = result[0][1]
                    pa_tem6 = pa_vec[5][i]
                    print(result[0][0], result[0][1])
    print('6')