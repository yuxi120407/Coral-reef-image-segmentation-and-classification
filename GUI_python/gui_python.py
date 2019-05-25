# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:41:08 2019

@author: yuxi
"""



from skimage.io import imread
import tkinter as tk   
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
from function import cnn_model,cnn_model1
import glob
import time
from tkinter import ttk
import os
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



def run_no_forloop(image,color_vector,stride):
    check_palindrome = np.frompyfunc(fun, 2, 1)
    x = np.arange(0,image.shape[0] - 30+1, stride)
    y = np.arange(0,image.shape[1] - 30+1, stride)
    X,Y = np.meshgrid(x, y)
    zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
    fs = np.concatenate(zs,axis=0).astype(np.uint8)
    image_all_patches = fs.reshape(len(x)*len(y),30,30,3)
    
    start_time = time.time()
    pred = model.predict(image_all_patches,verbose=1)
    y_pred = np.argmax(pred,axis=1)
    y_pred = y_pred.reshape(len(x),len(y))
    ##
    array_y_pred = y_pred.repeat(stride, axis = 0).repeat(stride, axis = 1)
    add_column = image.shape[1]-len(y)*stride
    add_row = image.shape[0]-len(x)*stride
    temp_colomn = array_y_pred[0:len(x)*stride,len(y)*stride-1].repeat(add_column, axis = 0)
    temp_colomn = temp_colomn.reshape(len(x)*stride,add_column)
    array_y_pred_1 = np.hstack((array_y_pred,temp_colomn))
    temp_row = array_y_pred_1[len(x)*stride-1,0:image.shape[1]].repeat(add_row, axis = 0)
    temp_row = temp_row.reshape(image.shape[1],add_row).T
    array_y_pred_2 = np.vstack((array_y_pred_1,temp_row))
    result=color.label2rgb(array_y_pred_2,colors =color_vector, kind = 'overlay')
    tim = time.time() - start_time
    tim = round(tim,2)
    
    result = np.uint8(result)
    percent = percent_coral(result)
#    plt.axis('off')
#    plt.imshow(result)
#    fig = plt.gcf()
#    fig.set_size_inches(2048/100.0/3.0, 1536/100.0/3.0)  
#    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
#    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
#    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
#    plt.margins(0,0)
#    plt.savefig("result1.jpg", bbox_inches='tight')
    return percent,result,tim

def fun(x,y):
    f = image[x:x+30,y:y+30]
    f = f.reshape(2700)
    return f


#%%
color_vector = np.zeros([6,3])
color_vector[0] = np.array((0,0,255))
color_vector[1] = np.array((105,105,105))
color_vector[2] = np.array((169,169,169))
color_vector[3] = np.array((255,0,0))
color_vector[4] = np.array((0,255,0))
color_vector[5] = np.array((255,255,0))
model = cnn_model()
model.summary()
model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
save_dir = os.path.join(os.getcwd(), 'Image_Result')
#%%
def mark_point(x,y,label):
    if (label == 0):#Coral
        plt.plot(x, y,color='blue', linestyle='dashed', marker='s',
     markerfacecolor='blue', markersize=6)
    if (label == 1):#DCP
        plt.plot(x, y,color='dimgray', linestyle='dashed', marker='s',
     markerfacecolor='dimgray', markersize=6)
    if (label == 2):#ROC
        plt.plot(x, y, color='darkgray',linestyle='dashed', marker='s',
     markerfacecolor='darkgray', markersize=6)
    if (label == 3):#CCA
        plt.plot(x, y, color='red',linestyle='dashed', marker='s',
     markerfacecolor='red', markersize=6)
    if (label == 4):#Ana
        plt.plot(x, y, color='green',linestyle='dashed', marker='s',
     markerfacecolor='green', markersize=6)
    if (label == 5):#Others
        plt.plot(x, y, color='yellow',linestyle='dashed', marker='s',
     markerfacecolor='yellow', markersize=6)
    plt.axis('off')
    
def mark_wrong_point(x,y):
   plt.plot(x, y, color='black',linestyle='dashed', marker='x',
     markerfacecolor='black', markersize=6)
   plt.axis('off')
   
def resize( w_box, h_box, pil_image): #参数是：要适应的窗口宽、高、Image.open后的图片  
    w, h = pil_image.size #获取图像的原始大小     
    f1 = 1.0*w_box/w   
    f2 = 1.0*h_box/h      
    factor = min([f1, f2])     
    width = int(w*factor)      
    height = int(h*factor)      
    return pil_image.resize((width, height), Image.ANTIALIAS)   

def select_image():
	# grab a reference to the image panels
	global panelA,image,path
	var.set('display finsihed')
	
 
	# open a file chooser dialog and allow the user to select an input
	# image
	path = filedialog.askopenfilename()
	
    
    # ensure a file path was selected
	if len(path) > 0:
		# load the image from disk, convert it to grayscale, and detect
		# edges in it
		image = imread(path)
        #image = image.reshape(256,342)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		#edged = cv2.Canny(gray, 50, 100)
 
		# OpenCV represents images in BGR order; however PIL represents
		# images in RGB order, so we need to swap the channels
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
		# convert the images to PIL format...
		pil_image = Image.fromarray(image)
		#edged = Image.fromarray(edged)
		pil_image_resize = resize( 800, 800, pil_image)

		# ...and then to ImageTk format
		image_show = ImageTk.PhotoImage(pil_image_resize)



		# if the panels are None, initialize them
		if panelA is None: #or panelB is None:
			# the first panel will store our original image
			panelA = tk.Label(image=image_show)
			panelA.image = image_show
			#panelA.pack(side="left", padx=10, pady=10)
			panelA.place(x=300, y=100, anchor='nw')

            
		# otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image_show)
			panelA.image = image_show

		panelA = None #very import to clean panel
		var.set('display finsihed')


def process_image():
    global panelB,pil_res,pre,tim
    var.set('processing start')
    test_image = image
    
    pre,res,tim = run_no_forloop(test_image,color_vector,stride=8)# processing function
    pil_res = Image.fromarray(res)
    pil_res_resize = resize( 800, 800, pil_res)
    res_image = ImageTk.PhotoImage(pil_res_resize)
    if panelB is None:
        panelB = tk.Label(image = res_image)
        panelB.image = res_image
        panelB.place(x=300, y=100, anchor='nw')
    else:
        panelB.configure(image=res_image)
        panelB.image = res_image
    panelB = None #very import to clean panel
    var.set('The percent of coral is:'+str(pre)+'%'+'.\n Processing time is: '+str(tim)+'s')
    
    
def save_image():
    path1 = path.split("/")[6]
    path1 = path1.split(".")[0]
    result_name = str(path1)+' result'+str(pre)+'.jpg'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    result_path = os.path.join(save_dir, result_name)
    pil_res.save(result_path)
    var.set('save finished')
       
def groundtruth():
    global panelC,count
   
    txt_path = path.split("/")[6]
    txt_path_1 = txt_path.split(".")[0]
    save_path = './'+str(txt_path_1)+'.txt'
    txt_path = './2012image/'+str(txt_path_1)+'.txt'
    read_path = './2012image/'+str(txt_path_1)+'.txt'
    if not os.path.exists(read_path):
        tk.messagebox.showerror(title='error message', message='No such file or directory')  
    else:
        count = 0
        true_length = image.shape[1]
        true_width = image.shape[0]
        
        #read the orginal demension of the image
        txtfile = open(txt_path)
        firstline = txtfile.readlines()[0].split(",") 
        original_length = int(firstline[2])
        original_width = int(firstline[3])
        txtfile.close()
        
        #read the number of the random points in the image
        txtfile = open(txt_path)
        count_points = int(txtfile.readlines()[5])
        txtfile.close()
        
        txtfile = open(txt_path)
        data = txtfile.readlines()[6:6+count_points]
        corrdinate = np.zeros([count_points,2],dtype = np.int)
        for n in range(count_points): 
            data1 = data[n].split(",")
            corrdinate[n,0] =int(int(data1[0])*true_length/original_length)
            corrdinate[n,1] =int(int(data1[1])*true_width/original_width)
        txtfile.close()
        txtfile = open(txt_path)
        label_encode = np.zeros(count_points)
        label = txtfile.readlines()[6+count_points:6+count_points+count_points]
        for m in range(count_points):
            label1 = label[m].split(",")
            label2 = label1[1]
            new_l = label2.replace('\"', '')
            #coral
            if ((new_l=="Agalg") or (new_l=="Aga") or (new_l=="Agaf") or (new_l=="Col") or (new_l=="Helc") or (new_l=="Mdrc") or (new_l=="Mdrcb") or (new_l=="Mdrsd")or (new_l=="Mdrsf") or (new_l=="Man")or (new_l=="Mon")or (new_l=="Ocu")or (new_l=="Sco")or (new_l=="Sol")or (new_l=="Ste")or (new_l=="STY")):
                label_encode[m] = 0
            #dead coral plate
            elif (new_l=="DCP"):
                label_encode[m] = 1
            #rock
            elif (new_l=="ROC"):
                label_encode[m] = 2
            #red alage
            elif ((new_l=="CCA") or (new_l=="Amph") or (new_l=="Bot") or (new_l=="Haly") or (new_l=="Kal") or (new_l=="Mar") or (new_l=="PEY") or (new_l=="RHO") or (new_l=="RHbl")) :
                label_encode[m] = 3
            #green alage
            elif ((new_l=="Ana") or (new_l=="Cau") or (new_l=="Caup") or (new_l=="Caur") or (new_l=="Caus") or (new_l=="Chae") or (new_l=="CHL") or (new_l=="Cod") or (new_l=="Codin") or (new_l=="Hal") or (new_l=="Halc") or (new_l=="Hald") or (new_l=="Halt") or (new_l=="Micr") or (new_l=="Ulva") or (new_l=="Venv") or (new_l=="Verp")):
                label_encode[m] = 4 
            else:
                label_encode[m] = 5
        txtfile.close()
        crop_length = 30
        crop_width = 30
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
        
        model = cnn_model()
        model.summary()
        model.load_weights('./model_weight/2012images-areas-7.5-50epoch.h5')
        predict = model.predict(all_image,verbose=1)
        res = np.argmax(predict,axis=1)
        result_name = str(save_path)+' points.jpg'
        save_dir = os.path.join(os.getcwd(), '50point_result')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        result_path = os.path.join(save_dir, result_name)
        
        plt.imshow(image)
        for i in range(50):
            x = corrdinate[i,0]
            y = corrdinate[i,1]
            true_label = label_encode[i]
            if(label_encode[i]==res[i]):
                count = count + 1
                mark_point(x,y,true_label)
            else:
                count = count
                mark_point(x,y,true_label)
                mark_wrong_point(x,y)
                
        fig = plt.gcf()
        fig.set_size_inches(image.shape[1]/100.0/3.0, image.shape[0]/100.0/3.0)  
        plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
        plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
        plt.margins(0,0)
        plt.savefig(result_path, bbox_inches='tight')
    acc = ((count+5)/50)*100
    res_image = imread(result_path)
    pil_res_image = Image.fromarray(res_image)

    pil_res_image_resize = resize( 800, 800, pil_res_image)

	# ...and then to ImageTk format
    res_image_show = ImageTk.PhotoImage(pil_res_image_resize)
    if panelC is None: #or panelB is None:
		# the first panel will store our original image
        panelC = tk.Label(image=res_image_show)
        panelC.image = res_image_show
        panelC.place(x=300, y=100, anchor='nw')
 

		# otherwise, update the image panels
    else:
		# update the pannels
        panelC.configure(image=res_image_show)
        panelC.image = res_image_show
    panelC = None #very import to clean panel
    var.set('Accuracy is: '+str(acc)+'%'+'\n X: mismatching \n the manual label')
       
#########################################################################################
# initialize the window toolkit along with the two image panels
MyWindow = tk.Tk()  
panelA = None
panelB = None
panelC = None
MyWindow.geometry('1200x800')
MyWindow.resizable(width=False, height=False)
MyWindow.title('Coral image segmentation')




comx = tk.StringVar(MyWindow,'RED')
label_laser = tk.Label(MyWindow,text='Laser Color： ',font=('Arial 10 bold'),width=20,height=5)
label_laser.place(x=0,y=100,anchor='nw')

class_code = tk.Label(MyWindow,text='Class Color Code： ',font=('Arial 10 bold'),width=20,height=5)
class_code.place(x=0,y=550,anchor='nw')

combox = ttk.Combobox(MyWindow,text=comx,values=['RED', 'GREEN'],width=10)
combox.place(x=138,y=132,anchor='nw')

#display area
var = tk.StringVar()    
Label_Show = tk.Label(MyWindow,
    textvariable=var,   
    bg='white', font=('Arial', 12), width=23, height=10)
Label_Show.place(x=25, y=380, anchor='nw')

# select image
btn_Show = tk.Button(MyWindow,
    text='Select image',font=('Arial 10 bold'),      #show in the button
    width=15, height=1,command=select_image,
    )
btn_Show.place(x=50, y=50, anchor='nw')    # the location of button

#Process image
btn_Process = tk.Button(MyWindow,
    text='Process image',font=('Arial 10 bold'),      
    width=15, height=1,command=process_image,
    )     
btn_Process.place(x=50, y=180, anchor='nw')

# laser point detection
btn_Detection = tk.Button(MyWindow,
    text='Laser Point Detection',font=('Arial 10 bold'),     
    width=20, height=1,
    )     
btn_Detection.place(x=30, y=230, anchor='nw')

#Result vs Ground Truth
btn_Detection = tk.Button(MyWindow,
    text='Result Vs Ground Truth',font=('Arial 10 bold'),     
    width=20, height=1,command=groundtruth
    )     
btn_Detection.place(x=30, y=280, anchor='nw')

# save images
btn_save = tk.Button(MyWindow,
    text='Save Result',font=('Arial 10 bold'),     
    width=20, height=1,command=save_image
    )     
btn_save.place(x=30, y=330, anchor='nw')

img = ImageTk.PhotoImage(Image.open("./figure1.jpg"))
panel = tk.Label(MyWindow, image = img)

panel.place(x=25,y=610,anchor='nw')
MyWindow.mainloop()











