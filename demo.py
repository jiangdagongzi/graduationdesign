
import tensorflow as tf
from inputdata import get_files, get_trains, get_tests, resultConversion, getMotionProfile

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