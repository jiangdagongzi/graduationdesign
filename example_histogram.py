# Author: Jean KOSSAIFI <jean.kossaifi@gmail.com>

from histogram import gradient, magnitude_orientation, hog, visualise_histogram
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg


def octagon(image_path):
     return cv2.imread(image_path, 0)


#   return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
#                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


img = octagon(r'C:\Users\A\Desktop\graduationdesign\train\has_train_101.PNG')
gx, gy = gradient(img, same_size=False)
mag, ori = magnitude_orientation(gx, gy)

# Show gradient and magnitude
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

# Show the orientation deducted from gradient
plt.figure()
plt.title('orientations')
plt.imshow(ori)
plt.pcolor(ori)
plt.colorbar()

# Plot histogram 
from scipy.ndimage.interpolation import zoom

# make the image bigger to compute the histogram
im1 = zoom(octagon('C:\\Users\\A\\Desktop\\graduationdesign\\train\\has_train_101.PNG'), 3)
h = hog(im1, cell_size=(4, 4), cells_per_block=(2, 2), visualise=False, nbins=4, signed_orientation=False,
        normalise=True)
im2 = visualise_histogram(h, 6, 6, False)

plt.figure()
plt.title('HOG features')
print(h.shape)
plt.imshow(h, cmap=plt.cm.Greys_r)
# print(len(im2.flatten()))

# plt.show()
# print(im2)
