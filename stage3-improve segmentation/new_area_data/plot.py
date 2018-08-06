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
path_txt = './2012_new/201208172_T-12-46-15_Dive_01_017.txt'
path_image = './201208172_T-12-58-58_Dive_01_041.jpg'
txtfile = open(path_txt)
firstline = txtfile.readlines()[2:] 
txtfile.close()
#%%
#Load Data
Data = np.loadtxt('./2012_new/201208172_T-12-46-15_Dive_01_017.txt')
#SpheresData = np.loadtxt('spheres.txt')
#SwissrollData = np.loadtxt('swissroll.txt')
#%%
#plot results
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(*[3,1,1], projection= '3d')
ax.scatter(EllipsoidsData[:,0],EllipsoidsData[:,1],EllipsoidsData[:,2])
myTitle = 'Elliposids'
ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))

#ax = fig.add_subplot(*[3,1,2], projection= '3d')
#ax.scatter(SpheresData[:,0],SpheresData[:,1],SpheresData[:,2])
#myTitle = 'SpheresData'
#ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
#
#ax = fig.add_subplot(*[3,1,3], projection= '3d')
#ax.scatter(SwissrollData[:,0],SwissrollData[:,1],SwissrollData[:,2])
#myTitle = 'SwissrollData'
#ax.set_title("\n".join(textwrap.wrap(myTitle, 20)))
