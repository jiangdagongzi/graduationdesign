# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:11:07 2018

@author: A
"""

import tensorflow as tf
from inputdata import get_files, get_trains, get_tests
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

# 初始化所有variables为0
sess.run(tf.initialize_all_variables())

# 构建回归模型
y_pre = tf.matmul(x, W) + b
y_pre_r = tf.nn.softmax(y_pre)

# 构造损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pre_r), axis=1))
cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_pre_r, 1e-10, 1.0)))
# cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_pre_r, 1e-10, 1.0)))

# 实现梯度下降
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer=tf.train.AdamOptimizer(1e-4).minimize(cost)

# 定义相关参数

# 训练循环次数
training_epochs = 100
# batch 一批，每次训练给算法10个数据
batch_size = 10
# 每隔5次，打印输出运算的结果
display_step = 5
# 定义训练数量
num_examples = 1500

# 预定义初始化
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
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
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print('epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Optimization Finished!')

    # 7.评估效果
    # Test model
correct_prediction = tf.equal(tf.argmax(y_pre_r, 1), tf.argmax(y, 1))
# Calculate accuracy for 3000 examples
# tf.cast类型转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
img_list_test, lab_list_test = get_tests()
print("Accuracy:", accuracy.eval({x: img_list_test, y: lab_list_test}))
#    print("Accuracy:",accuracy.eval({X: mnist.test.images[:3000], Y: mnist.test.labels[:3000]}))
