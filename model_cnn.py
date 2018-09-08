# -*- coding: utf-8 -*-
"""
Created on Fri May 25 10:27:31 2018

@author: A
"""

import tensorflow as tf
from inputdata import get_files, get_trains, get_tests, get_trains_hog, get_tests_hog, resultConversion, \
    getMotionProfile, resultConversion_CNN
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据的存储位置
# train_dir = 'C:\\Users\\A\\Desktop\\train\\'#your data directory

# 获取session
sess = tf.InteractiveSession()

# 创建两个占位符，因为图片是48*48的，所以维度是2304
# 识别结果只有两种，所以长度是2
x = tf.placeholder("float", shape=[None, 2304])
y = tf.placeholder("float", shape=[None, 2])

##定义相关参数


# 训练循环次数
training_epochs = 50
# batch 一批，每次训练给算法10个数据
batch_size = 10
# 每隔5次，打印输出运算的结果
display_step = 5
# 定义训练数量
num_examples = 130


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 24])
b_conv1 = bias_variable([24])

x_image = tf.reshape(x, [-1, 48, 48, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 24, 48])
b_conv2 = bias_variable([48])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([12 * 12 * 48, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 48])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(y_)
# 定义损失函数
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), axis=1))
cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
# cost = tf.reduce_sum(tf.square(tf.substract(y_conv, y)))
# cost = -tf.reduce_sum(y_conv * tf.log(y))
# learning rate
learning_rate = 0.001
# 优化器
train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
# train_step = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(cost)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 初始化所有参数
sess.run(tf.initialize_all_variables())

for epoch in range(training_epochs):
    avg_cost = 0.
    # 总训练批次total_batch =训练总样本量/每批次样本数量
    total_batch = int(num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = get_trains(batch_size)
        # a = sess.run(y_conv, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
        # b = sess.run(W_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
        # c = sess.run(W_conv2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
        # d = sess.run(b_conv1, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
        # e = sess.run(b_conv2, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})
        _, _c = sess.run([train_step, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        avg_cost += _c / total_batch
    if (epoch + 1) % display_step == 0:
        print('epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

# saver = tf.train.Saver()
# saver.save(sess, 'saved_model_cnn/model_cnn.ckpt')

# 评估
test_x, test_y = get_tests()
poss = sess.run(y_conv, feed_dict={x: test_x, y: test_y, keep_prob: 1.0})
print(poss)
y_conv_result = resultConversion_CNN(poss)
# correct_prediction = tf.equal(tf.argmax(y_conv_result, 1), tf.argmax(y, 1))
correct_prediction = tf.equal(tf.argmax(y_conv_result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_x, y: test_y, keep_prob: 1.0}))

