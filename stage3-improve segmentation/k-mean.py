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









