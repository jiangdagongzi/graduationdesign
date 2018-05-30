# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:21:36 2018

@author: A
"""
import cv2
from histogram import gradient, magnitude_orientation, hog, visualise_histogram
import matplotlib.pyplot as plt

def octagon(image_path):
   return cv2.imread(image_path,0)

def read(image_path):  
    img = octagon(image_path)
    gx, gy = gradient(img, same_size=False)
    mag, ori = magnitude_orientation(gx, gy)
    return img

def showGradient(img,gx,gy,mag):
    plt.figure()
    plt.title('gradients and magnitude')
    plt.subplot(141)
    plt.imshow(img, cmap=plt.cm.Greys_r)
    plt.subplot(142)
    plt.imshow(gx, cmap=plt.cm.Greys_r)
    plt.subplot(143)
    plt.imshow(gy, cmap=plt.cm.Greys_r)
    plt.subplot(144)
    plt.imshow(mag, cmap=plt.cm.Greys_r)
    return plt
    
def showOrientation(ori):
    plt.figure()
    plt.title('orientations')
    plt.imshow(ori)
    plt.pcolor(ori)
    plt.colorbar()
    return plt
    
def showHistogram(image_path):
    from scipy.ndimage.interpolation import zoom
    # make the image bigger to compute the histogram
    im1 = zoom(octagon(image_path), 3)
    h = hog(im1, cell_size=(8, 8), cells_per_block=(2, 2), visualise=False, nbins=6, signed_orientation=False, normalise=True)
    im2 = visualise_histogram(h, 6, 6, False)
#    plt.imshow(im2, cmap=plt.cm.Greys_r)
    return im2


#from scipy.ndimage.interpolation import zoom
#    # make the image bigger to compute the histogram
#im = octagon('C:\\Users\\A\\Desktop\\train\\has_train_1.PNG')
#im2 = showHistogram('C:\\Users\\A\\Desktop\\train\\has_train_1.PNG')
#print(len(im2.flatten()))
#print(len(im1.flatten()))

  