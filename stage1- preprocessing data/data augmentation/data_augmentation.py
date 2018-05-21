# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:47:33 2018

@author: yuxi
"""

from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import glob
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import os  



def data_augmentation(ClassName,ImageNum):
    className = str(ClassName)
    imageNum = ImageNum -1
    datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                                 shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
    image_files = glob.glob("./data/"+className+"/*.jpg")
    for name in image_files:
        name_image = name.split("\\")[1]
        image_subidex = name_image.split('.')[0]
        img = load_img(name)  
        x = img_to_array(img)  # transfer image to numpy, the shape of it is (31, 31, 3)
        x = x.reshape((1,) + x.shape)  # transfer shape to (1,31, 31, 3)
        i = 0 
        for batch in datagen.flow(x,batch_size=1,save_to_dir="data/"+className,save_prefix=image_subidex+"-New", save_format = "jpg"):
            i = i+1
            if(i>imageNum):
                break
    
    

#%% save crop image in folder
def save_crop_image(path_txt,path_image,image_subidex):
    image_idex = str(image_subidex)
    corrdinate = np.zeros([50,2],dtype = np.int)
    all_image = np.zeros([50,30,30,3],dtype=np.uint8)
    label = np.zeros(50,dtype = np.int)
    txtfile = open(path_txt)
    data = txtfile.readlines()[2:52]
    image = imread(path_image)
    for n in range(50):
        data1 = data[n].split(",")
        label[n] = int(data1[2])
        idex = str(label[n])
        corrdinate[n,0] =int(data1[0])
        corrdinate[n,1] =int(data1[1])
        if(corrdinate[n,0]-15 <0):
            corrdinate[n,0] = 15
        elif(corrdinate[n,1]-15 <0):
            corrdinate[n,1] = 15
        elif(corrdinate[n,0]+15 >2048):
            corrdinate[n,0] = 2048-15
        elif(corrdinate[n,1]+15 >1536):
            corrdinate[n,1] =1536-15
        all_image[n,:,:,:] = image[corrdinate[n,1]-15:corrdinate[n,1]+15,corrdinate[n,0]-15:corrdinate[n,0]+15]
        im = Image.fromarray(all_image[n,:,:,:])
        if (label[n]==0):
            idex = str(n)
            im.save("./data/Coral/Coral-"+image_idex+"-"+idex+".jpg")
        if (label[n]==1):
            idex = str(n)
            im.save("./data/DCP/DCP-"+image_idex+"-"+idex+".jpg")
        if (label[n]==2):
            idex = str(n)
            im.save("./data/Rock/Rock-"+image_idex+"-"+idex+".jpg")
        if (label[n]==3):
            idex = str(n)
            im.save("./data/Red algae/Red algae-"+image_idex+"-"+idex+".jpg")
        if (label[n]==4):
            idex = str(n)
            im.save("./data/Green algae/Green algae-"+image_idex+"-"+idex+".jpg")
        if (label[n]==5):
            idex = str(n)
            im.save("./data/Others/Others-"+image_idex+"-"+idex+".jpg")
#%%  
read_files = glob.glob("./2012new/*.txt")
for name in read_files:
        image_idex = name.split("_")[4]
        image_subidex = image_idex.split(".")[0]
        name = name.split("\\")[1]
        name = name.split(".")[0]
        path_txt = str('./2012new/')+name+str('.txt')
        path_image = str('./image/')+name+str('.jpg')
        save_crop_image(path_txt,path_image,image_subidex)
#%%
image_label = np.zeros(1)
image_data = np.zeros([1,30,30,3])
read_files = glob.glob('./data/Coral/*.jpg')  
for name in read_files:
    name_path = name.split("\\")[1]
    name_idex = name_path.split(".")[0]
    img_label = name_idex.split("-")[0]
    
    im = imread(name)
    im = im.reshape((1,) + im.shape)  # transfer shape to (1,31, 31, 3)
    image_data = np.vstack((image_data,im))
    if(img_label=="Coral"):
        new_label = 1
    image_label = np.hstack((image_label,new_label))
    final_image_data = image_data[1:,:,:,:]
    final_label = image_label[1:]
#%%
def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]
    
dirs = os.listdir("./data")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
