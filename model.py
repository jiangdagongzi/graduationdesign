# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:11:07 2018

@author: A
"""

import tensorflow as tf
from inputdata import get_files, get_trains, get_tests, resultConversion, getMotionProfile
import numpy as np
import matplotlib.pyplot as plt

# 数据的存储位置
# train_dir = 'C:\\Users\\A\\Desktop\\train\\'#your data directory

# 创建session
sess = tf.InteractiveSession()

# 创建两个占位符，因为图片是48*48的，所以维度是2304
# 识别结果只有两种，所以长度是2
x = tf.placeholder("float", shape=[None, 2304])
y = tf.placeholder("float", shape=[None, 2])

# 构建权重矩阵
W = tf.Variable(tf.zeros([2304, 2]))
# 构建偏置矩阵
b = tf.Variable(tf.zeros([2]))

# 构建回归模型
y_pre = tf.matmul(x, W) + b
y_pre_r = tf.nn.softmax(y_pre)

# 构造损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre_r), axis=1))
cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_pre_r, 1e-10, 1.0)))
# cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_pre_r, 1e-10, 1.0)))

# 实现梯度下降
learning_rate = 0.01
optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

# optimizer=tf.train.AdamOptimizer(1e-4).minimize(cost)


# 定义相关参数

# 训练循环次数
training_epochs = 120
# batch 一批，每次训练给算法10个数据
batch_size = 10
# 每隔5次，打印输出运算的结果
display_step = 5
# 定义训练数量
num_examples = 1300

# 预定义初始化
init = tf.global_variables_initializer()

# 初始化
sess.run(init)
# 循环训练次数
#    imagepath,batch_ys = get_files(train_dir)

for epoch in range(training_epochs):
    avg_cost = 0.
    # 总训练批次total_batch =训练总样本量/每批次样本数量
    total_batch = int(num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = get_trains(batch_size)
        # print(batch_xs[0])
        # plt.imshow(batch_xs[0].reshape([48, 48]), cmap=plt.cm.Greys_r)
        a = sess.run(W, feed_dict={x: batch_xs, y: batch_ys})
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += c / total_batch
    if (epoch + 1) % display_step == 0:
        print('epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

print('Optimization Finished!')

saver = tf.train.Saver()
saver.save(sess, 'saved_model/model.ckpt')

img_list_test, lab_list_test = get_tests()

# print(sess.run(accuracy, feed_dict={x: img_list_test, y: lab_list_test}))
poss = sess.run(y_pre_r, feed_dict={x: img_list_test, y: lab_list_test})
print(poss)
y_pre_r_result = resultConversion(poss)

# print(y_pre_r_result)
correct_prediction = tf.equal(tf.argmax(y_pre_r_result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: img_list_test, y: lab_list_test}))


