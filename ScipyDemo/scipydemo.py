# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:31:45 2019

@author: liguanlin
"""
import numpy as np
from scipy.misc import imread, imsave, imresize
import matplotlib.pyplot as plt

img = imread('assets/thumbs_up.jpg')

print(img.dtype, img.shape)


img_tinted = img * [1, 0.95, 0.9]

img_tinted = imresize(img_tinted, (300, 300))

imsave('assets/thumbs_up_tinted.jpg', img_tinted)

#show the original image
plt.subplot(1,2,1)
plt.imshow(img)

#show the tinted image
plt.subplot(1,2,2)
plt.imshow(np.uint8(img_tinted))

plt.show()