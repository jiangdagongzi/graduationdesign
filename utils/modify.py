# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:04:45 2018

@author: A
"""
from PIL import Image  
import os.path  
import glob  
def convertjpg(jpgfile,outdir,width=48,height=48):  
    img=Image.open(jpgfile)  
    try:  
        new_img=img.resize((width,height),Image.BILINEAR)     
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))  
    except Exception as e:  
        print(e)  
for jpgfile in glob.glob("C:\\Users\\A\\Desktop\\3\\*.PNG"):  
    convertjpg(jpgfile,"C:\\Users\\A\\Desktop\\3\\")