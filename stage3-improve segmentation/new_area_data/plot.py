# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:43:16 2018

@author: yuxi
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
import math 
from mpl_toolkits.mplot3d import Axes3D
import textwrap
import glob
from skimage.io import imread  
#%%
txt_name = glob.glob("./2012_new/*.txt")
data = np.zeros([1,3])
for name in txt_name: 
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    path_txt = str('./2012_new/')+image_name+str('.txt')
    path_image =  str('./2012image/')+image_name+str('.jpg')

    txtfile = open(path_txt)
    firstline = txtfile.readlines()[2:] 
    txtfile.close()
    image = imread(path_image)
    for i in range(50):
        all_data = np.zeros([50,3])
        test_file = firstline[i].split(",")
        point_x = int(test_file[0])
        point_y = int(test_file[1])
        label = int(test_file[2])
        all_data[i,:]=image[point_y,point_x]
    data = np.vstack((data,all_data))
real_data = data[1:]

#plot results
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(*[1,1,1], projection= '3d')
ax.scatter(real_data[:,0],real_data[:,1],real_data[:,2])
#ax.scatter(0,0,255,color='red')
myTitle = 'source domain'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

