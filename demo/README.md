## Coral image segmentation and classification

### Step1: install the dependence in the terminal:
pip install tensorflow. 
pip install keras.

### Step2: generate data with K-means
"generate_data_with K-means.py" is used to do label augmentation with K_menas.

### Step3: read data and transfer crop image data
you can get the label augmentation data of the 2012 images in the "2012data_label augmentation(m=300)"floder.
and run the "function.py", you can find the function to generate data and transfer the data into what we need to train with the CNN model in this python file, 
other funciton like plot learning curve and confusion matrix are also in this file.

### Step4: train with CNN
"crop_image_CNN.py" is used to train our data with CNN, we have already save the weight which we have trained in the "model_weight " floder.

### Step5: do the image segmentation and classification 
"test_image.py" is used to do the image segmentation and classificaiton and calculate the coral percent in the images.

When you run the "test_image.py", you should put the image you want to test in the "test" file and the result image will be 
saved in the "result_image" file.

"2012image" contains raw images and origianl label data txt format files
"2012new" contains the result of the processing data which we can read
"2012data_label augmentation(m=300)" contains the label augmentation data
"model_weight" contains the weights which we have already trained
"result_image" contains the result of the image segmentation
"test" contains the image you want to test
"test_image" contains all the 2012images 
