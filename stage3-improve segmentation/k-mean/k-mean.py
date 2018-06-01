# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:24:10 2018

@author: yuxi
"""


# import the necessary packages
from skimage.segmentation import slic
from skimage.data import astronaut
import matplotlib.pyplot as plt
from skimage.io import imread
img = astronaut()
segments = slic(img, n_segments=100, compactness=10)


#%%
path_image = '201208172_T-12-58-58_Dive_01_041.jpg'
test_image = imread(path_image)
d = slic(test_image, n_segments=100, compactness=10.0, max_iter=10, sigma=0, spacing=None, multichannel=True, convert2lab=True, enforce_connectivity=True, min_size_factor=0.5, max_size_factor=3, slic_zero=False)
plt.imshow(d)
#%%
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
path_image = "./2012image/201208172_T-12-50-10_Dive_01_025.jpg"
#img = img_as_float(astronaut()[::2, ::2])
img = imread(path_image)
img = img_as_float(img)
#%%image segmentation with k-means

#segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=15, sigma=1)
#segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
#gradient = sobel(rgb2gray(img))
#segments_watershed = watershed(gradient, markers=250, compactness=0.001)

#print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
#print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
#plot the boundary of the SLIC algorithm
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)

#ax[0, 0].imshow(mark_boundaries(img, segments_fz))
#ax[0, 0].set_title("Felzenszwalbs's method")
ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
ax.set_title('SLIC')
#ax[1, 0].imshow(mark_boundaries(img, segments_quick))
#ax[1, 0].set_title('Quickshift')
#ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
#ax[1, 1].set_title('Compact watershed')

#for a in ax.ravel():
    #a.set_axis_off()

plt.tight_layout()
plt.show()
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
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
#mark_point(1530,35,2)
#plt.plot(300,300,c='rx')



count_points = 50
geneate_count_points = 500
path_txt = "./2012new/201208172_T-12-50-10_Dive_01_025.txt"
txtfile = open(path_txt)
all_lines = txtfile.readlines()[2:]
lines = all_lines[0:50]
generate_lines = all_lines[50:]
for i in range(count_points):
    line_piece = lines[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    mark_point(l_x,l_y,original_label)
for j in range(geneate_count_points):
    line_piece = generate_lines[j]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    mark_generate_point(l_x,l_y,original_label)    
















































#%%generate samples: generate 10 points for each original points
def generate_sample(original_label,label,size,write_path):
    a = np.where(segments_slic==label)
    a_y = a[0]
    a_x = a[1]
    num_dime = a_x.shape[0]
    f = np.random.randint(num_dime, size=size)
    corrdinate_x = np.zeros(size)
    corrdinate_y = np.zeros(size)
    for i,j in enumerate (f):
        corrdinate_x[i] = a_x[j]
        corrdinate_y[i] = a_y[j]
        writ = open(write_path,"a") #writ is a file object for write
        x = int(corrdinate_x[i])
        y = int(corrdinate_y[i])
        label = original_label
        writ.write('{0},{1},{2}\n'.format(x,y,label))
        writ.close()
#%% generate 550 points for each txt documents
count_points = 50
path_txt = "./2012new/201208172_T-12-50-10_Dive_01_025.txt"
txtfile = open(path_txt)
lines = txtfile.readlines()[2:52]
for i in range(count_points):
    line_piece = lines[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    label = segments_slic[l_y,l_x]
    generate_sample(original_label,label,10,path_txt)
txtfile.close()
#%%
def generate_signal_imagedata(path_txt,path_image):
    #path_txt = "./2012new/201208172_T-12-58-58_Dive_01_041.txt"
    txt_file = open(path_txt)
    text = txt_file.readlines()[2:]
    #path_image = "./201208172_T-12-58-58_Dive_01_041.jpg"
    count_points = 550
    crop_length = 30
    crop_width = 30
    all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
    label = np.zeros(count_points)
    crop_x = int(crop_length/2)
    crop_y = int(crop_width/2)
    image = imread(path_image)
    for i in range(count_points):
        text_piece = text[i]
        text_element = text_piece.split(',')
        l_x = int(text_element[0])
        l_y = int(text_element[1])
        label[i] = int(text_element[2])
        if(l_x-crop_x <0):
            l_x = crop_x
        if(l_y-crop_y <0):
            l_y = crop_y
        if(l_x+crop_x >2048):
            l_x = 2048-15
        if(l_y+crop_y >1536):
            l_y =1536-15
        all_image[i,:,:,:] = image[l_y-15:l_y+15,l_x-15:l_x+15]
    txt_file.close()
    return all_image,label
#%%
count_points = 50
raw_txt_files = glob.glob("./2012new/*.txt")
count = 0
for txt_name in raw_txt_files:
    start = time.time()
    name = txt_name.split("\\")[1]
    name = name.split(".")[0]
    path_image = str('./2012image/')+name+str('.jpg')
    txt_file = open(txt_name)
    text = txt_file.readlines()[2:]
    img = imread(path_image)
    img = img_as_float(img)
    segments_slic = slic(img, n_segments=100, compactness=0.1, sigma=5)
     
    for i in range(count_points): 
       line_piece = text[i]
       list_element = line_piece.split(',')
       l_x = int(list_element[0])
       l_y = int(list_element[1])
       original_label = int(list_element[2])
       label = segments_slic[l_y,l_x]
       generate_sample(original_label,label,10,txt_name)
    txt_file.close()
    end = time.time()
    count = count+1
    print(("---image%d finished in %s seconds ---" % (count,(end - start))))
#%%
all_label = np.zeros(1)
image_data = np.zeros([1,30,30,3],dtype=np.uint8)
read_files = glob.glob("./2012new/*.txt")
for name in read_files:
    name = name.split("\\")[1]
    name = name.split(".")[0]
    path_txt = str('./2012new/')+name+str('.txt')
    path_image = str('./2012image/')+name+str('.jpg')
    new_image_data,label = generate_signal_imagedata(path_txt,path_image)
    all_label = np.hstack((all_label,label))
    image_data = np.vstack((image_data,new_image_data))
final_data = image_data[1:,:,:,:]
final_label = all_label[1:]
#%%
































