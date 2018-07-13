
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 14:16:48 2018

@author: yuxi
"""

# import the necessary packages
import keras
from skimage.segmentation import slic
from skimage.data import astronaut
import matplotlib.pyplot as plt
from skimage.io import imread
import pylab

import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import time

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float
#%%
def mark_point(x,y,label):
    if (label == 0):#Coral
        plt.plot(x,y,'bs')
    if (label == 1):#DCP
        plt.plot(x,y,'ks')
    if (label == 2):#ROC
        plt.plot(x,y,'kx')
    if (label == 3):#CCA
        plt.plot(x,y,'rs')
    if (label == 4):#Ana
        plt.plot(x,y,'gs')
    if (label == 5):#Others
        plt.plot(x,y,'ms')
    plt.axis('off')
    

def mark_generate_point(x,y,label):
    if (label == 0):#Coral
        plt.plot(x,y,'b^')
    if (label == 1):#DCP
        plt.plot(x,y,'k^')
    if (label == 2):#ROC
        plt.plot(x,y,'k^')
    if (label == 3):#CCA
        plt.plot(x,y,'r^')
    if (label == 4):#Ana
        plt.plot(x,y,'g^')
    if (label == 5):#Others
        plt.plot(x,y,'m^')
    plt.axis('off')
#%%
def mergedata(image_name):
    #image_name = '201208172_T-12-58-58_Dive_01_041'
    area_path_txt = str('./2012new - Copy/')+image_name+str('.txt')
    write_path = str('./2012new/')+image_name+str('.txt')
    area_txtfile = open(area_path_txt) 
    area_lines = area_txtfile.readlines()[2:]
    area_txtfile.close()
    size_area = len(area_lines)
    for i in range(size_area):
        area_lines_piece = area_lines[i]
        area_list_element = area_lines_piece.split(',')
        writ = open(write_path,"a") #writ is a file object for write
        area_x = int(float(area_list_element[0]))
        area_y = int(float(area_list_element[1]))
        area_label = int(area_list_element[2])
        writ.write('{0},{1},{2}\n'.format(area_x,area_y,area_label))
        writ.close()
#%%
image_name = glob.glob("./2012new - Copy/*.txt")
count = 1
for name in image_name:
    img_name = name.split("\\")[1]
    img_name = img_name.split(".")[0]
    mergedata(img_name)
    print("Image {0} is finish" .format(count))
    count = count + 1
#%%
imageName = "201208172_T-12-46-15_Dive_01_017"
path_image = str('./test_image/')+imageName+str('.jpg')
img = imread(path_image)
img = img_as_float(img)
segments_slic = slic(img, n_segments=300, compactness=15, sigma=1)
write_path = str('./2012new/')+imageName+str('.txt')
area_txtfile = open(write_path) 
area_lines = area_txtfile.readlines()[2:]
area_txtfile.close()
size_area = len(area_lines)
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
for i in range(size_area):
    area_lines_piece = area_lines[i]
    area_list_element = area_lines_piece.split(',')
    area_x = int(area_list_element[0])
    area_y = int(area_list_element[1])
    area_label = int(area_list_element[2])
    mark_point(area_x,area_y,area_label)
   


#%%read original pixels labels
def generateLabel(imageName):
#imageName = "201208172_T-14-18-29_Dive_01_118"
    path_image = str('./test_image/')+imageName+str('.jpg')
    img = imread(path_image)
    img = img_as_float(img)
    segments_slic = slic(img, n_segments=300, compactness=15, sigma=1)
    label_image = np.zeros([1536,2048])
    path_txt = str('./2012new/')+imageName+str('.txt')
    txtfile = open(path_txt) 
    lines = txtfile.readlines()[2:]
    txtfile.close()
    count_points = len(lines)
    for i in range(count_points):
        line_piece = lines[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        label = segments_slic[l_y,l_x]
        a = np.where(segments_slic==label)
        a_y = a[0]
        a_x = a[1]
        zipped = zip(a_y,a_x)
        for m,n in zipped:
            label_image[m,n]=original_label+1
    return label_image
#%%
label= np.zeros([126,1536,2048])
imageName = glob.glob("./2012new/*.txt")
#imageName = imageName[90:]
count_generateLabel = 1
for m,name in enumerate(imageName):
    img_Name = name.split("\\")[1]
    img_Name = img_Name.split(".")[0]
    start_time = time.time()
    label[m] = generateLabel(img_Name)
    print("Image %d is finished in %s second" %(count_generateLabel,(time.time()-start_time)))
    count_generateLabel = count_generateLabel + 1

#%%save label
im_name = imageName[90:]
count_saveLabel = 1
for n,name in enumerate(im_name):
    im_name = name.split("\\")[1]
    im_name = im_name.split(".")[0]
    label_path = str('./label/')+im_name+str('.txt')
    g = label[n]
    np.savetxt(label_path,g,fmt='%d')
    print("Image {0} is finish" .format(count_saveLabel))
    count_saveLabel = count_saveLabel +1
    
#%%show the image segmentaion result
ground_truth = np.zeros([1536,2048,3])
for i in range(1536):
    for j in range(2048):
        la = int(data[i,j])
        if(la == 1):#coral---blue
            ground_truth[i,j]=np.array([0,0,255])
        elif(la == 2): #DCP---slight gray
            ground_truth[i,j]=np.array([105,105,105])
        elif(la == 3): #ROC---deep gray
            ground_truth[i,j]=np.array([169,169,169])
        elif(la == 4): #CCA---red
            ground_truth[i,j]=np.array([255,0,0])
        elif(la == 5): #Ana---green
            ground_truth[i,j]=np.array([0,255,0])
        elif(la == 6):
            ground_truth[i,j]=np.array([255,255,0])
        else:
            ground_truth[i,j]=np.array([0,0,0])
ground_truth = ground_truth.astype(np.uint8)
#%%
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
for i in range(count_points):
    line_piece = lines[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    mark_point(l_x,l_y,original_label)
#plt.savefig('./2012image_label_visual/'+image_name+'original_points.png',bbox_inches='tight')
#%%generate weight of each images
def generate_weight_mask(original_imglabel_path):
#original_imglabel_path = './label/201208172_T-12-46-15_Dive_01_017.txt'
    original_imglabel = np.loadtxt(original_imglabel_path,dtype = np.uint8) 
    weight_mask = np.zeros([1536,2048],dtype =np.uint8)
    for i in range(1536):
        for j in range(2048):
            lab = original_imglabel[i,j]
            if ((lab==1) or (lab==2) or (lab==3) or (lab==4) or (lab==5) or (lab==6)):
                weight_mask[i,j] = 1
            else:
                weight_mask[i,j] = 0
    return weight_mask
#%%
imageName = glob.glob("./2012new/*.txt")
for m,name in enumerate(imageName):
    img_Name = name.split("\\")[1]
    img_Name = img_Name.split(".")[0]
    label_path = str('./label/')+img_Name+str('.txt')
    weight_mask = generate_weight_mask(label_path)
    weight_mask_path = str('./weight_mask/')+img_Name+str('.txt')
    np.savetxt(weight_mask_path,weight_mask,fmt='%d')




























