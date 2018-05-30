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

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

#img = img_as_float(astronaut()[::2, ::2])
img = imread(path_image)

segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=250, compactness=0.001)

print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

fig, ax = plt.subplots(2, 2, figsize=(5, 5), sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()



#%%----------------------------------------------------------------------------------------------------
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

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

#img = img_as_float(astronaut()[::2, ::2])
img = imread(path_image)
img = img_as_float(img)
#%%

#segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=100, compactness=30, sigma=1)
#segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
#gradient = sobel(rgb2gray(img))
#segments_watershed = watershed(gradient, markers=250, compactness=0.001)

#print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
#print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
#%%
fig, ax = plt.subplots(1, 1, figsize=(10,5), sharex=True, sharey=True)

#ax[0, 0].imshow(mark_boundaries(img, segments_fz))
#ax[0, 0].set_title("Felzenszwalbs's method")
ax.imshow(mark_boundaries(img, d,color=(1,1,0)))
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
def generate_sample(original_label,label,size):
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
        writ = open("./201208172_T-12-58-58_Dive_01_041.txt","a") #writ is a file object for write
        x = int(corrdinate_x[i])
        y = int(corrdinate_y[i])
        label = original_label
        writ.write('{0},{1},{2}\n'.format(x,y,label))
        writ.close()
#%%
count_points = 50
path_txt = "./2012new/201208172_T-12-58-58_Dive_01_041.txt"
txtfile = open(path_txt)
lines = txtfile.readlines()[2:52]
for i in range(count_points):
    line_piece = lines[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    label = segments_slic[l_y,l_x]
    generate_sample(original_label,label,10)
txtfile.close()
#%%
path_txt = "./201208172_T-12-58-58_Dive_01_041.txt"
txt_file = open(path_txt)
text = txt_file.readlines()[2:]

#%%
path_image = "./201208172_T-12-58-58_Dive_01_041.jpg"
count_points = 500
crop_length = 30
crop_width = 30
all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
crop_x = int(crop_length/2)
crop_y = int(crop_width/2)
image = imread(path_image)
for i in range(count_points):
    text_piece = text[i]
    text_element = text_piece.split(',')
    l_x = int(text_element[0])
    l_y = int(text_element[1])
    if(l_x-crop_x <0):
        l_x = crop_x
    elif(l_y-crop_y <0):
        l_y = crop_y
    elif(l_x+crop_x >2048):
        l_x = 2048-15
    elif(l_y+crop_y >1536):
        l_y =1536-15
    all_image[i,:,:,:] = image[l_y-15:l_y+15,l_x-15:l_x+15]








