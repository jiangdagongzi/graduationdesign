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
from design import showHistogram

train_dir = 'C:\\Users\\A\\Desktop\\graduationdesign\\train\\'
test_dir = 'C:\\Users\\A\\Desktop\\graduationdesign\\test\\'

# 定义训练数量
train_num = 1500
# 定义测试数量
test_num = 250


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
    print('There are %d hasnot\nThere are %d has' % (len(iris), len(contact)))

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
