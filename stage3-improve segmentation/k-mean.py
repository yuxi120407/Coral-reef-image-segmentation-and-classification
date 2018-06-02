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
#%%generate new augmentation txt files
count_points = 50
raw_txt_files = glob.glob("./2012new/*.txt")
for txt_name in raw_txt_files:
     txt_file = open(txt_name)
     text = txt_file.readlines()[2:]
     for i in range(count_points): 
        line_piece = text[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        label = segments_slic[l_y,l_x]
        generate_sample(original_label,label,10,txt_name)
     txt_file.close()
#%%
def drew_label_points(image_name):
    #image_name = str()
    path_image = str('./2012image/')+image_name+str('.jpg')
    path_txt = str('./2012new - No_augmentation/')+image_name+str('.txt')
    print("find points by using k-means... ")
    img = imread(path_image)
    img = img_as_float(img)
    segments_slic = slic(img, n_segments=50, compactness=15, sigma=1)
    print("k-means done")
    print("generate 10 points for each label...")
    count_points = 50
    txtfile = open(path_txt)
    lines = txtfile.readlines()[2:52]
    for i in range(count_points):
        line_piece = lines[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        label = segments_slic[l_y,l_x]
        generate_sample(segments_slic,original_label,label,10,path_txt)
    txtfile.close()
    print("geneating done")
    geneate_count_points = 500
    txtfile = open(path_txt)
    all_lines = txtfile.readlines()[2:]
    lines = all_lines[0:50]
    generate_lines = all_lines[50:]
    fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
    ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
    for i in range(count_points):
        line_piece = lines[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        mark_point(l_x,l_y,original_label)
    plt.savefig('./2012-test-image/'+image_name+'original_points.png',bbox_inches='tight')
    for j in range(geneate_count_points):
        line_piece = generate_lines[j]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        mark_generate_point(l_x,l_y,original_label)   
    plt.savefig('./2012-test-image/'+image_name+'generate_points.png',bbox_inches='tight')
#%%
image_count = 120
count = 1
txt_name = raw_txt_files = glob.glob("./2012new/*.txt")
for name in txt_name:
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    print("Image {0} is processing ".format(count))
    start_time = time.time()
    drew_label_points(image_name)
    #print("Image {0} finished ".format(count))
    print(("Image %d finished in %f seconds ---" % (count,(time.time()-start_time))))
    count = count+1






