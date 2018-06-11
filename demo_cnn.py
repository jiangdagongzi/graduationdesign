import tensorflow as tf
from inputdata import getMotionProfile, resultConversion_CNN_demo,getMotionProfile_demo,resultConversion_CNN
import cv2

test_PNG = r'C:\Users\A\Desktop\graduationdesign\test_34.PNG'

# 声明模型

x = tf.placeholder("float", shape=[None, 2304])


# y = tf.placeholder("float", shape=[None, 2])


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


def result_box(image_path, x_stride, y_stride, poss):
    poss = poss.flatten()
    im = cv2.imread(image_path, 0)
    y, x = im.shape
    xnum = (x - 48) // x_stride + 1
    ynum = (y - 48) // y_stride + 1
    for i in range(len(poss) // 2):
        if poss[2 * i + 1] == 1:
            cv2.rectangle(im, ((i % xnum) * x_stride, (i // xnum) * y_stride),
                          ((i % xnum) * x_stride + 48, (i // xnum) * y_stride + 48), (0, 255, 0), 4)
    cv2.imwrite(test_PNG, im)


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

# 获取图片信息
# batch_mp = getMotionProfile(r'C:\Users\A\Desktop\graduationdesign\test.PNG')
batch_mp = getMotionProfile_demo(test_PNG)

saver = tf.train.Saver()

with tf.Session() as sess:
    # 加载训练好的模型graph
    saver = tf.train.import_meta_graph('saved_model_cnn/model_cnn.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('saved_model_cnn/'))

    b = sess.run(W_fc2)
    poss = sess.run(y_conv, feed_dict={x: batch_mp, keep_prob: 1.0})

    y_result = resultConversion_CNN(poss)

    # result_box(r'C:\Users\A\Desktop\graduationdesign\test.PNG', 16, 8, y_result)
    result_box(test_PNG, 48, 48, y_result)
