# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:22:07 2019

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
from PIL import Image
from function import cnn_model,cnn_model1
import glob
import time
#%%
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    f = np.where(test[:,:,2]==255)
    count = len(f[0])
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent

def run_no_forloop(test_image,color_vector,file_image,name_image,stride):
    check_palindrome = np.frompyfunc(fun, 2, 1)
    x = np.arange(0,test_image.shape[0] - 30+1, stride)
    y = np.arange(0,test_image.shape[1] - 30+1, stride)
    X,Y = np.meshgrid(x, y)
    zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
    fs = np.concatenate(zs,axis=0).astype(np.uint8)
    image_all_patches = fs.reshape(len(x)*len(y),30,30,3)
    pred = model.predict(image_all_patches,verbose=1)
    y_pred = np.argmax(pred,axis=1)
    y_pred = y_pred.reshape(len(x),len(y))
    ## add value in two side
    array_y_pred = y_pred.repeat(stride, axis = 0).repeat(stride, axis = 1)
    add_column = test_image.shape[1]-len(y)*stride
    add_row = test_image.shape[0]-len(x)*stride
    temp_colomn = array_y_pred[0:len(x)*stride,len(y)*stride-1].repeat(add_column, axis = 0)
    temp_colomn = temp_colomn.reshape(len(x)*stride,add_column)
    array_y_pred_1 = np.hstack((array_y_pred,temp_colomn))
    temp_row = array_y_pred_1[len(x)*stride-1,0:test_image.shape[1]].repeat(add_row, axis = 0)
    temp_row = temp_row.reshape(test_image.shape[1],add_row).T
    array_y_pred_2 = np.vstack((array_y_pred_1,temp_row))
    result=color.label2rgb(array_y_pred_2,colors =color_vector, kind = 'overlay')
    result = np.uint8(result)
    percent = percent_coral(result)
    plt.axis('off')
    plt.imshow(result)
    fig = plt.gcf()
    fig.set_size_inches(test_image.shape[1]/100.0/3.0, test_image.shape[0]/100.0/3.0)  
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.savefig('./all_new_image/'+str(file_image)+'/'+str(name_image)+' result'+str(percent)+'.pdf', bbox_inches='tight')
    plt.close()
    return percent

def fun(x,y):
    f = test_image[x:x+30,y:y+30]
    f = f.reshape(2700)
    return f
#%%
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
#%%
color_vector = np.zeros([6,3])
color_vector[0] = np.array((0,0,255))
color_vector[1] = np.array((105,105,105))
color_vector[2] = np.array((169,169,169))
color_vector[3] = np.array((255,0,0))
color_vector[4] = np.array((0,255,0))
color_vector[5] = np.array((255,255,0))
#%%
path_image = './2012image/201208172_T-12-49-32_Dive_01_024.jpg'
test_image = imread(path_image)
count = 1
start_time = time.time()
percent = run_no_forloop(test_image,color_vector)
print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
print("there are {0}% coral in this image".format(percent))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#run all images
stride = 8
first_list  = glob.glob("./all_new_image/*")
for file in first_list:
    file_image = file.split("\\")[1]
    raw_image_files = glob.glob('./all_new_image/'+str(file_image)+'/*.jpg')
    count = 0
    for name in raw_image_files:
        name_piece = name.split("\\")[1]
        name_image = name_piece.split(".")[0]
        path_image = './all_new_image/'+str(file_image)+'/'+str(name_image)+'.jpg'
        testimage = Image.open(path_image)
        test_image = imread(path_image)
        start_time = time.time()
        percent = run_no_forloop(test_image,color_vector,file_image,name_image,stride)
        count = count+1
        print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
        print("there are {0}% coral in this image".format(percent))
    print("--------------------images in the {} finished -------------".format(file_image) )





































