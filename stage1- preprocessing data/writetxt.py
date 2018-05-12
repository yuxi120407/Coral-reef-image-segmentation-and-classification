# -*- coding: utf-8 -*-
"""
Created on Sat May 12 13:08:16 2018

@author: yuxi
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import glob


im = Image.open('201208172_T-12-58-58_Dive_01_041.jpg') #relative path to file
 
#load the pixel info
pix = im.load()
 
#get a tuple of the x and y dimensions of the image
width, height = im.size
 
#open a file to write the pixel data
with open('output_file.txt', 'w+') as f:
  f.write('R,G,B\n')
 
  #read the details of each pixel and write them to the file
  for x in range(width):
    for y in range(height):
      r = pix[x,y][0]
      g = pix[x,y][1]
      b = pix[x,y][2]
      f.write('{0},{1},{2}\n'.format(r,g,b))

#%%
for filename in glob.glob('*.txt'):
    f = open(filename, 'r')
      