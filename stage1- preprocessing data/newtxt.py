# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:36:24 2018

@author: Xi Yu
"""

def newtxt(path_txt,path_image):
    txtfile = open(path_txt)
    firstline = txtfile.readlines()[0].split(",") 
    original_length = int(firstline[2])
    original_width = int(firstline[3])
    count_points = int(txtfile.readlines()[5])
    data = txtfile.readlines()[6:6+count_points]
    corrdinate = np.zeros([count_points,2],dtype = np.int)
    for n in range(count_points): 
        data1 = data[n].split(",")
        corrdinate[n,0] =int(int(data1[0])*2048/original_length)
        corrdinate[n,1] =int(int(data1[1])*1536/original_width)
     label_encode = np.zeros(count_points)
     label = txtfile.readlines()[56+count_points:6+count_points+count_points]
     for m in range(count_points):
         label1 = label[m].split(",")
         label2 = label1[1]
         new_l = label2.replace('\"', '')
         if (new_l=="Agalg"):
             label_encode[m] = 0
         if (new_l=="DCP"):
             label_encode[m] = 1
         if (new_l=="ROC"):
             label_encode[m] = 2
         if (new_l=="TWS"):
             label_encode[m] = 3
         if (new_l=="CCA"):
             label_encode[m] = 4
         if (new_l=="Davi"):
             label_encode[m] = 5
         if (new_l=="Ana"):
             label_encode[m] = 6  
         else:
             label_encode[m] = 7
       txtfile.close()
       name_image = path_image.split("/")[2] 
       name_txt = name_image.split(".")[0]
       with open(str('new/')+name_txt+str('.txt'), 'w+') as newtxtfile:
           imagefile.write(name_image+str('\n'))
           imagefile.write('x,y,label\n')
           for i in range(50):
               x = corrdinate[i,0]
               y = corrdinate[i,1]
               label = int(label_encode[i])
               newtxtfile.write('{0},{1},{2}\n'.format(x,y,label))
           newtxtfile.close()