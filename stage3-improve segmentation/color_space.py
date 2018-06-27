# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:15:23 2018

@author: Xi Yu
"""

from skimage import color
from skimage import data
from matplotlib import pyplot as plt
from skimage.io import imread
from PIL import Image
import cv2
import numpy as np
#%%
def rgb2ycbcr(im_rgb):
 im_rgb = im_rgb.astype(np.float32)
 im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
 im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
 im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0 #to [16/255, 235/255]
 im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
 return im_ycbcr
#%%
path_img = "./201208172_T-12-58-58_Dive_01_041.jpg"
img = imread(path_img)
img = img/255
img_hsv = color.rgb2hsv(img)
img_lab = color.rgb2lab(img)
img_hed = color.rgb2hed(img)
img_rgbcie = color.rgb2rgbcie(img)
img_xyz = color.rgb2xyz(img)
img_yuv = color.rgb2yuv(img)
img_yiq = color.rgb2yiq(img)
img_ycbcr = color.rgb2ycbcr(img)
img1 = color.ycbcr2rgb(img_ycbcr)
#%%


#%%
fig, ax = plt.subplots(2, 4, figsize=(15,8))
ax[0, 0].imshow(img)
ax[0, 0].set_title("RGB image")
ax[0, 1].imshow(img_hsv)
ax[0, 1].set_title("HSV image")
ax[0, 2].imshow(img_lab)
ax[0, 2].set_title('LAB image')
ax[0, 3].imshow(img_hed)
ax[0, 3].set_title('HED image')
ax[1, 0].imshow(img_xyz)
ax[1, 0].set_title('XYZ image')
ax[1, 1].imshow(img_yuv)
ax[1, 1].set_title('yuv image')
ax[1, 2].imshow(img_yiq)
ax[1, 2].set_title('LCH image')
ax[1, 3].imshow(img_ycbcr)
ax[1, 3].set_title('YCBCR image')