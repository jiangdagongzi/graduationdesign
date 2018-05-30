# -*- coding: utf-8 -*-
"""
Created on Mon May 21 15:28:19 2018

@author: A
"""

import os
path='C:\\Users\\A\\Desktop\\1\\'      

#获取该目录下所有文件，存入列表中
f=os.listdir(path)

n=0
for i in f:
    
    #设置旧文件名（就是路径+文件名）
    oldname=path+f[n]
    
    #设置新文件名
    newname=path+'has_'+'train_'+str(n+1)+'.PNG'
    
    #用os模块中的rename方法对文件改名
    os.rename(oldname,newname)
    print(oldname,'======>',newname)
    
    n+=1