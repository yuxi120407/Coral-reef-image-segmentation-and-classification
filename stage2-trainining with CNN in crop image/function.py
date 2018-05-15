# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:34:27 2018

@author: yuxi
"""

import matplotlib.pyplot as plt
import numpy as np 
from skimage.io import imread
from PIL import Image
import glob
import itertools
from sklearn.metrics import classification_report, confusion_matrix

def newtxt(path_txt,path_image,crop_length,crop_width):
    # load the true demension of the image
    image = imread(path_image)
    true_length = image.shape[1]
    true_width = image.shape[0]
    
    #read the orginal demension of the image
    txtfile = open(path_txt)
    firstline = txtfile.readlines()[0].split(",") 
    original_length = int(firstline[2])
    original_width = int(firstline[3])
    txtfile.close()
    
    #read the number of the random points in the image
    txtfile = open(path_txt)
    count_points = int(txtfile.readlines()[5])
    txtfile.close()
    
    #read the corrdinate of the image
    txtfile = open(path_txt)
    data = txtfile.readlines()[6:6+count_points]
    corrdinate = np.zeros([count_points,2],dtype = np.int)
    for n in range(count_points): 
        data1 = data[n].split(",")
        corrdinate[n,0] =int(int(data1[0])*true_length/original_length)
        corrdinate[n,1] =int(int(data1[1])*true_width/original_width)
    txtfile.close()
    
    #read the label of each points and encode the label
    txtfile = open(path_txt)
    label_encode = np.zeros(count_points)
    label = txtfile.readlines()[6+count_points:6+count_points+count_points]
    for m in range(count_points):
        label1 = label[m].split(",")
        label2 = label1[1]
        new_l = label2.replace('\"', '')
        if (new_l=="Agalg"):
            label_encode[m] = 0
        elif (new_l=="DCP"):
            label_encode[m] = 1
        elif (new_l=="ROC"):
            label_encode[m] = 2
        elif (new_l=="TWS"):
            label_encode[m] = 3
        elif (new_l=="CCA"):
            label_encode[m] = 4
        elif (new_l=="Davi"):
            label_encode[m] = 5
        elif (new_l=="Ana"):
            label_encode[m] = 6  
        else:
            label_encode[m] = 7
    txtfile.close()
    name_image = path_image.split("/")[2] 
    name_txt = name_image.split(".")[0]
    with open(str('2012new/')+name_txt+str('.txt'), 'w+') as newtxtfile:
        newtxtfile.write(name_image+str('\n'))
        newtxtfile.write('x,y,label\n')
        for i in range(count_points):
            x = corrdinate[i,0]
            y = corrdinate[i,1]
            label = int(label_encode[i])
            newtxtfile.write('{0},{1},{2}\n'.format(x,y,label))
        newtxtfile.close()
    all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
    crop_x = int(crop_length/2)
    crop_y = int(crop_width/2)
    for i in range(count_points):
        if(corrdinate[i,0]-crop_x <0):
            corrdinate[i,0] = crop_x
        elif(corrdinate[i,1]-crop_y <0):
            corrdinate[i,1] = crop_y
        elif(corrdinate[i,0]+crop_x >true_length):
            corrdinate[i,0] = true_length-crop_x
        elif(corrdinate[i,1]+crop_y >true_width):
            corrdinate[i,1] =true_width-crop_y
        all_image[i,:,:,:] = image[corrdinate[i,1]-crop_y:corrdinate[i,1]+crop_y,corrdinate[i,0]-crop_x:corrdinate[i,0]+crop_x]
    return all_image, label_encode
#%%
def newimagedata(crop_length,crop_width):
    all_label = np.zeros(1)
    image_data = np.zeros([1,crop_length,crop_width,3],dtype=np.uint8)
    read_files = glob.glob("./2012image/*.txt")
    #length = len(read_files)
    for name in read_files:
        name = name.split("\\")[1]
        name = name.split(".")[0]
        path_txt = str('./2012image/')+name+str('.txt')
        path_image = str('./2012image/')+name+str('.jpg')
        new_image_data,label = newtxt(path_txt,path_image,crop_length,crop_width)
        all_label = np.hstack((all_label,label))
        image_data = np.vstack((image_data,new_image_data))
    final_data = image_data[1:,:,:,:]
    final_label = all_label[1:]
    return final_data,final_label
#%%
def create_plots(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of CNN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('accuracy.png')
    #plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('margin_loss of CNN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig('loss.png')
    #plt.clf()
    
def plot_confusion_matrix(confusionmatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusionmatrix)

    plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, format(confusionmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')









