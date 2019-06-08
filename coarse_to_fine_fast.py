# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 22:43:34 2018

@author: yuxi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image,ImageDraw
from function import cnn_model,cnn_model1,cnn_model2
from skimage.measure import block_reduce
from sklearn.cluster import KMeans
import glob
import time
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from skimage import color
#%%find the precent areas of coral
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    pixel = np.zeros([1,1,3],dtype = np.uint8)
    for r in range(0,test.shape[0]):
        for c in range(0,test.shape[1]):
            pixel = test[r,c,:]
            #if(np.all(pixel == blue)):
            if((pixel[0]==0)and(pixel[1]==0)and(pixel[2]==255)):
                count = count +1
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent

#%%test the image
def test_img_loc(img,model):
    windowsize_r = 30
    windowsize_c = 30
    count = 0
    #test_image = img_new
    testimage = Image.new('RGB', (683,512))
    result = np.zeros([510,690,3],dtype=np.uint8)
    result[:,0:683,:] = img[0:510,:]
    start_time = time.time()
    loc_x = []
    loc_y = []
    for r in range(0,result.shape[0] - windowsize_r+1, 30):
            for c in range(0,result.shape[1] - windowsize_c+1, 30):
                window = result[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = model.predict(window,verbose=2)
                y_pred = np.argmax(pred,axis=1)
                if(y_pred==0):#coral---blue
                    testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
                    loc_x = np.append(loc_x,r+15)
                    loc_y = np.append(loc_y,c+15) 
                    
                elif(y_pred==1):#DCP---slight gray
                    testimage.paste((105,105,105),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==2):#ROC---deep gray
                    testimage.paste((169,169,169),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==3):#CCA---red
                    testimage.paste((255,0,0),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==4):#Ana---green
                    testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
                else:#others---yellow
                    testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])
    count = count+1
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    plt.imshow(testimage)
    plt.axis('off')
    plt.savefig('35.pdf')
    return loc_x,loc_y
#%%
def test_img_label(img,model,img_label):
    start_time = time.time()
    windowsize_r = 30
    windowsize_c = 30
    count = 0
    testimage = Image.new('RGB', (img.shape[1],img.shape[0]))
    for r in range(0,img.shape[0] - windowsize_r+1, 6):
        for c in range(0,img.shape[1] - windowsize_c+1, 6):
            if(image_label[r,c]==1):
                window = img[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = detail_model.predict(window,verbose=2)
                y_pred = np.argmax(pred,axis=1)
                if(y_pred==0):#coral---blue
                    testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==1):#DCP---slight gray
                    testimage.paste((105,105,105),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==2):#ROC---deep gray
                    testimage.paste((169,169,169),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==3):#CCA---red
                    testimage.paste((255,0,0),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==4):#Ana---green
                    testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
                else:#others---yellow
                    testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])
    count = count+1    
    percent = percent_coral(testimage)
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    print("there are {0}% coral in this image".format(percent))
    plt.imshow(testimage)
    plt.axis('off')
    testimage.save('./New folder/result'+str(percent)+'.pdf')
#%%
#get the area of the original image
def get_back_img(img,center,dis,factor):
    area_x1 = factor*np.int(center[0]-dis-15)
    area_x2 = factor*np.int(center[0]+dis+15)
    area_y1 = factor*np.int(center[1]-dis-15)
    area_y2 = factor*np.int(center[1]+dis+15)
    
    if(area_x1<0):
        area_x1 = 0
    if(area_x2>img.shape[0]):
        area_x2 = img.shape[0]
    if(area_y1<0):
        area_y1 = 0
    if(area_y2>img.shape[1]):
        area_y2 = img.shape[1]
    back_img = img[area_x1:area_x2,area_y1:area_y2]
    plt.imshow(np.uint8(back_img))
    return back_img,area_x1,area_x2,area_y1,area_y2
#get the length of the area
def get_dis(data,center):
    temp_dis = np.zeros(len(data))
    for i in range(len(data)):
        temp = center-data[i,:]
        temp_value = temp@temp.T
        temp_dis[i] =  np.sqrt(temp_value)
    dis = np.max(temp_dis)
    return dis
#plot the rectangle in the original image
def plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2):
    #draw = ImageDraw.Draw(im)
    draw.line((area_y1, area_x1, area_y1, area_x2), fill="blue", width=10)
    draw.line((area_y1, area_x2, area_y2, area_x2), fill="blue", width=10)
    draw.line((area_y2, area_x2, area_y2, area_x1), fill="blue", width=10)
    draw.line((area_y2, area_x1, area_y1, area_x1), fill="blue", width=10)
#%%#%%load coarse model and weight
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2018_10_14_weight0.2.h5')
#%%
#step1 downsample image
path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
img = imread(path_image)
k = img[0:1536:3,0:2048:3,:]
plt.imshow(k)
plt.axis('off')
#%%
#step2 coarse search the downsampe image
loc_x,loc_y = test_img_loc(k,model)
#%%store the location of the coral
coral_loc = np.vstack((loc_x,loc_y)).T
image_label = np.zeros([img.shape[0],img.shape[1]]) 
size = len(coral_loc)
im = Image.open(path_image)
img = imread(path_image)
draw = ImageDraw.Draw(im)
for i in range(size):
    _,area_x1,area_x2,area_y1,area_y2 = get_back_img(img,coral_loc[i,:],20,3)
    image_label[area_x1:area_x2,area_y1:area_y2] = 1
    plot_rectangle(draw,area_x1,area_x2,area_y1,area_y2)
im.save('test11.pdf')   
plt.imshow(image_label,cmap='Greys_r')
plt.axis('off') 
#%%
detail_model = cnn_model()
detail_model.summary()
detail_model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
#%%
test_img_label(img,detail_model,image_label)


#%%

path_image = "./2012image/201208172_T-12-56-18_Dive_01_036.jpg"
img = imread(path_image)
plt.imshow(img)
plt.axis('off')
plt.savefig('36.pdf')



#%% fast way to do coarse-to-fine
#step1 resample image

path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
img = imread(path_image)
k = img[0:1536:3,0:2048:3,:]
plt.imshow(k)
plt.axis('off')
#%%
def fun(x,y):
    f = test_image[x:x+30,y:y+30]
    f = f.reshape(2700)
    return f

def run_no_forloop(test_image,color_vector,stride):#file_image,name_image,stride):
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
    #h = np.where(y_pred==0)
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
#    percent = percent_coral(result)
#    plt.axis('off')
#    plt.imshow(result)
#    fig = plt.gcf()
#    fig.set_size_inches(test_image.shape[1]/100.0/3.0, test_image.shape[0]/100.0/3.0)  
#    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
#    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
#    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
#    plt.margins(0,0)
#    #the address to save the result
#    plt.savefig('./result_image/'+'/'+str(name_image)+' result'+str(percent)+'.pdf', bbox_inches='tight')
#    plt.close()
    return result

#%%
start_time = time.time()
color_vector = np.zeros([7,3])
color_vector[0] = np.array((0,0,0))
color_vector[1] = np.array((0,0,255))
color_vector[2] = np.array((105,105,105))
color_vector[3] = np.array((169,169,169))
color_vector[4] = np.array((255,0,0))
color_vector[5] = np.array((0,255,0))
color_vector[6] = np.array((255,255,0))
#step1 dowmsample the image
path_image = "./2012image/201208172_T-12-54-11_Dive_01_032.jpg"
img = imread(path_image)
test_image = img[0:1536:3,0:2048:3,:]
plt.imshow(test_image)
plt.axis('off')
##step2 get the results of downsample images with lower threshold CNN
stride = 30
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
y_pred = np.uint8(y_pred)
##step3 mapping back at original image
factor = 3
dis = 20
g = np.where(y_pred==0)
local_x = g[0]*30 + 15
local_y = g[1]*30 + 15
are_x1 = factor*(local_x-dis-15)
are_x1[are_x1<0]=0
are_x2 = factor*(local_x+dis+15)
are_x2[are_x2>img.shape[0]]=img.shape[0]
are_y1 = factor*(local_y-dis-15)
are_y1[are_y1<0]=0
are_y2 = factor*(local_y+dis+15)
are_y2[are_y2>img.shape[1]]=img.shape[1]
length = len(are_y2)
image_label = np.zeros([img.shape[0],img.shape[1]]) 
for i in range(length):
    image_label[are_x1[i]:are_x2[i],are_y1[i]:are_y2[i]] = 1
plt.imshow(image_label,cmap='Greys_r')
##only search areas in the potential areas
nu_pa = np.ones([1,30,30,3])*300
nu_pa = nu_pa.reshape(2700)
def new_fun(x,y):
    if image_label[x,y]==1:
        m = img[x:x+30,y:y+30]
        m = m.reshape(2700)
        return m
    else:
        return nu_pa               
check_palindrome = np.frompyfunc(new_fun, 2, 1)
stride = 6
x = np.arange(0,img.shape[0] - 30+1, stride)
y = np.arange(0,img.shape[1] - 30+1, stride)
X,Y = np.meshgrid(x, y)
zs= check_palindrome(np.ravel(X.T), np.ravel(Y.T))
fs = np.concatenate(zs,axis=0).astype(np.uint16)
ff = fs[fs!=300]
image_some_patches = ff.reshape(-1,30,30,3)
pred_some = detail_model.predict(image_some_patches,verbose=1)
y_pred_some = np.argmax(pred_some,axis=1)
j = np.arange(0,229294800,2700)
ffs = fs[j]
gg = np.where(ffs!=300)
gg = gg[0]
index_x = gg//337
index_y = gg%337
size_some_patch = len(index_x)
re_pre = np.zeros([252,337])
for i in range (size_some_patch):
    re_pre[index_x[i],index_y[i]] = y_pred_some[i]+1
re_pre = re_pre.repeat(stride, axis = 0).repeat(stride, axis = 1)
result=color.label2rgb(re_pre,colors =color_vector, kind = 'overlay')
result = np.uint8(result)
plt.imshow(result)
print('the time of coarse_to_fine is: {:.4f}'.format(time.time()-start_time))



