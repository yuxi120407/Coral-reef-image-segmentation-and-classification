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

#%%
import sys,os,msvcrt  #导入的模块与方法
   
 def join(in_filenames, out_filename):  
     out_file = open(out_filename, 'w+')  
       
     err_files = []  
     for file in in_filenames:  
         try:  
             in_file = open(file, 'r')  
             out_file.write(in_file.read())  
             out_file.write('\n\n')  
             in_file.close()  
         except IOError:  
             print 'error joining', file  
             err_files.append(file)  
     out_file.close()  
     
     print 'joining completed. %d file(s) missed.' % len(err_files)  
     
     print 'output file:', out_filename  
     
     if len(err_files) > 0:  #判断
         print 'missed files:'  
         print '--------------------------------'  
         for file in err_files:  
             print file  
         print '--------------------------------'  
 #www.iplaypy.com  
 if __name__ == '__main__':  
     print 'scanning...'  
     in_filenames = []  
     file_count = 0  
     
     for file in os.listdir(sys.path[0]):  
         if file.lower().endswith('[all].txt'):  
             os.remove(file)  
         elif file.lower().endswith('.txt'):  
             in_filenames.append(file)  
             file_count = file_count + 1  
     
     if len(in_filenames) > 0:  
         print '--------------------------------'  
         print '\n'.join(in_filenames)  
         print '--------------------------------'  
         print '%d part(s) in total.' % file_count  
         book_name = raw_input('enter the book name: ')  
         print 'joining...'  
         join(in_filenames, book_name + '[ALL].TXT')  
     else:  
         print 'nothing found.'  
     
     msvcrt.getch() 
      
