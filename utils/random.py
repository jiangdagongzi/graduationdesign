# -*- coding: utf-8 -*-
"""
Created on Wed May 16 16:10:24 2018

@author: A
"""
import random
import os
import shutil


def random_copyfile(srcPath, dstPath, numfiles):
    name_list = list(os.path.join(srcPath, name) for name in os.listdir(srcPath))
    random_name_list = list(random.sample(name_list, numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.copyfile(oldname, oldname.replace(srcPath, dstPath))


srcPath = 'F:\\学习\\大四下\\毕设\\切割图\\'
dstPath = 'C:\\Users\\A\\Desktop\\3\\'
random_copyfile(srcPath, dstPath, 800)
