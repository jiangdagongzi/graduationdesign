# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:14:11 2018

@author: A
"""

import random
import numpy as np
import os
import cv2
import tensorflow as tf
from PIL import Image
from design import showHistogram
import glob

train_dir = 'C:\\Users\\A\\Desktop\\graduationdesign\\train\\'
test_dir = 'C:\\Users\\A\\Desktop\\graduationdesign\\test3\\'
mp_dir = 'C:\\Users\\A\\Desktop\\graduationdesign\\mp_test\\'
mp_dir_glob = 'C:\\Users\\A\\Desktop\\graduationdesign\\mp_test\\*.PNG'

# 定义训练数量
train_num = 1300
# 定义测试数量
test_num = 200


def octagon(image_path):
    return cv2.imread(image_path, 0)


def octagon1(iamge_path):
    return tf.gfile.FastGFile(image_list, 'rb')


def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    iris = []
    label_iris = []
    contact = []
    label_contact = []
    for file in os.listdir(file_dir):
        name = file.split('_')
        if name[0] == "hasnot":
            iris.append(file_dir + file)
            label_iris.append(0)
        else:
            contact.append(file_dir + file)
            label_contact.append(1)
    # print('There are %d hasnot\nThere are %d has' % (len(iris), len(contact)))

    image_list = np.hstack((iris, contact))
    label_list = np.hstack((label_iris, label_contact))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    #    print(image_list)
    #    print(label_list)
    return image_list, label_list


image_list, label_list = get_files(train_dir)


def get_trains(size):
    '''
    args:
        image_list :images paths
        label_list :labels paths
        size :how many a time
    '''

    img_list = np.empty(shape=[0, 2304])
    lab_list = np.empty(shape=[0, 2])
    for i in range(size):
        #        print(image_list[start+i])
        rand = random.randint(0, train_num - 1)
        img_list = np.vstack((img_list, octagon(image_list[rand]).flatten()))
        if label_list[rand] == 0:
            lab_list = np.vstack((lab_list, np.array([1, 0], dtype=np.float)))
        else:
            lab_list = np.vstack((lab_list, np.array([0, 1], dtype=np.float)))
    img_list.reshape(size, 2304)
    lab_list.reshape(size, 2)
    return img_list, lab_list


def get_trains_hog(size):
    img_list_hog = np.empty(shape=[0, 10404])
    lab_list_hog = np.empty(shape=[0, 2])

    for i in range(size):
        rand = random.randint(0, train_num - 1)
        img = showHistogram(image_list[rand])
        img_list_hog = np.vstack((img_list_hog, img.flatten()))
        if label_list[rand] == 0:
            lab_list_hog = np.vstack((lab_list_hog, np.array([1, 0], dtype=np.float)))
        else:
            lab_list_hog = np.vstack((lab_list_hog, np.array([0, 1], dtype=np.float)))
    return img_list_hog, lab_list_hog


image_list_test, label_list_test = get_files(test_dir)


def get_tests():
    img_list_test = np.empty(shape=[0, 2304])
    lab_list_test = np.empty(shape=[0, 2])
    for i in range(len(image_list_test)):
        img_list_test = np.vstack((img_list_test, octagon(image_list_test[i]).flatten()))
        if label_list_test[i] == 0:
            lab_list_test = np.vstack((lab_list_test, np.array([1, 0], dtype=np.float)))
        else:
            lab_list_test = np.vstack((lab_list_test, np.array([0, 1], dtype=np.float)))
    img_list_test.reshape(test_num, 2304)
    lab_list_test.reshape(test_num, 2)
    return img_list_test, lab_list_test


def get_tests_hog():
    img_list_test_hog = np.empty(shape=[0, 10404])
    lab_list_test_hog = np.empty(shape=[0, 2])
    for i in range(len(image_list_test)):
        img = showHistogram(image_list_test[i])
        img_list_test_hog = np.vstack((img_list_test_hog, img.flatten()))
        if label_list_test[i] == 0:
            lab_list_test_hog = np.vstack((lab_list_test_hog, np.array([1, 0], dtype=np.float)))
        else:
            lab_list_test_hog = np.vstack((lab_list_test_hog, np.array([0, 1], dtype=np.float)))
    img_list_test_hog.reshape(test_num, 10404)
    lab_list_test_hog.reshape(test_num, 2)
    return img_list_test_hog, lab_list_test_hog


# 定义结果转换
def resultConversion(y_poss):
    y_result = np.empty(shape=[0, 2])
    y_poss = y_poss.flatten()
    for i in range(y_poss.shape[0] // 2):
        if y_poss[2 * i] > 0.6:
            y_result = np.vstack((y_result, np.array([1, 0], dtype=np.float)))
        else:
            y_result = np.vstack((y_result, np.array([0, 1], dtype=np.float)))
    return y_result


# 定义结果转换
def resultConversion_CNN(y_poss):
    y_result = np.empty(shape=[0, 2])
    y_poss = y_poss.flatten()
    for i in range(y_poss.shape[0] // 2):
        if y_poss[2 * i] > 0.46:
            y_result = np.vstack((y_result, np.array([1, 0], dtype=np.float)))
        else:
            y_result = np.vstack((y_result, np.array([0, 1], dtype=np.float)))
    return y_result


# 定义结果转换
def resultConversion_CNN_demo(y_poss):
    y_result = np.empty(shape=[0, 2])
    y_poss = y_poss.flatten()
    for i in range(y_poss.shape[0] // 2):
        if y_poss[2 * i] > 0.457:
            y_result = np.vstack((y_result, np.array([1, 0], dtype=np.float)))
        else:
            y_result = np.vstack((y_result, np.array([0, 1], dtype=np.float)))
    return y_result

# 获取mp文件测试信息
# def getMotionProfile(image_path):
#     im = cv2.imread(image_path, 0)
#     x, y = im.shape
#     xnum = x // 48
#     ynum = y // 48
#     mp_image = np.empty(shape=[0, 2304])
#     mp_label = np.empty(shape=[0, 2])
#     for c in range(xnum):
#         for r in range(ynum):
#             mp_image = np.vstack((mp_image, im[c * 48:48, r * 48:48].flatten()))
#             mp_label = np.vstack((mp_label, np.array([1, 0], dtype=np.float)))
#     return mp_image, mp_label


# mp_image, mp_label = getMotionProfile(r'C:\Users\A\Desktop\graduationdesign\test.png')
# print(len(mp_image))
# print(mp_image[0].shape)


# def getMotionProfile(image_path):
#     im = cv2.imread(image_path, 0)
#     x, y = im.shape
#     xnum = x // 48
#     ynum = y // 48
#     splitimage(image_path, xnum, ynum, mp_dir)
#     mp_image = np.empty(shape=[0, 2304])
#     mp_label = np.empty(shape=[0, 2])
#     for pngfile in glob.glob(mp_dir_glob):
#         mp_image = np.vstack((mp_image, octagon(pngfile).flatten()))
#         mp_label =np.vstack((mp_label, np.array([0, 1], dtype=np.float)))
#     return mp_image,mp_label

# def splitimage(src, rownum, colnum, dstpath):
#     img = Image.open(src)
#     w, h = img.size
#     if rownum <= h and colnum <= w:
#         print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
#         print('开始处理图片切割, 请稍候...')
#
#         s = os.path.split(src)
#         if dstpath == '':
#             dstpath = s[0]
#         fn = s[1].split('.')
#         basename = fn[0]
#         ext = fn[-1]
#
#         num = 0
#         rowheight = h // rownum
#         colwidth = w // colnum
#         for r in range(rownum):
#             for c in range(colnum):
#                 box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
#                 img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
#                 num = num + 1
#
#         print('图片切割完毕，共生成 %s 张小图片。' % num)
#     else:
#         print('不合法的行列切割参数！')

# 定义扫描步长,以像素为单位
x_stride = 16
y_stride = 8


# 获取MP文件
def getMotionProfile(image_path):
    im = cv2.imread(image_path, 0)
    y, x = im.shape
    xnum = (x - 48) // 16 + 1
    ynum = (y - 48) // 8 + 1
    mp_list = np.empty(shape=[0, 2304])
    for i in range(ynum):
        for j in range(xnum):
            aim = im[(i * 8):(i * 8 + 48), (j * 16):(j * 16 + 48)]
            mp_list = np.vstack((mp_list, aim.flatten()))
            # print(i, j)
    return mp_list

def getMotionProfile_demo(image_path):
    im = cv2.imread(image_path, 0)
    y, x = im.shape
    xnum = x//48
    ynum = y//48
    mp_list = np.empty(shape=[0, 2304])
    for i in range(ynum):
        for j in range(xnum):
            aim = im[(i * 48):(i * 48 + 48), (j * 48):(j * 48 + 48)]
            mp_list = np.vstack((mp_list, aim.flatten()))
            # print(i, j)
    return mp_list
