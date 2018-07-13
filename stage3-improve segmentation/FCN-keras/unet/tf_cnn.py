# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:48:19 2018

@author: Xi Yu
"""
#%%
import numpy as np
import os
import struct
import matplotlib
import matplotlib.pyplot as plt  
import math 
import textwrap
from array import array as pyarray

import tensorflow as tf
#%%
Dataset_path = './MNIST_data/'
def load_mnist(dataset="training", digits=np.arange(10), path= Dataset_path, size = 60000):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-label-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = size #int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N): #int(len(ind) * size/100.)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])\
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels, rows, cols
#%%
training_images, training_labels,training_rows, training_cols  = load_mnist('training')
testing_images, testing_labels,testing_rows, testing_cols  = load_mnist('testing')

train_label = np.array(training_labels)
test_label = np.array(testing_labels)

n_images = len(training_images)
n_labels = len(training_labels)

test_n_images = len(testing_images)
test_n_labels = len(testing_labels)
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
train_image = training_images.reshape((n_images, -1))
train_labels = train_label.reshape((n_labels, -1))
test_image = testing_images.reshape((test_n_images, -1))
test_labels = test_label.reshape((test_n_labels, -1))

features = train_image[:-1]
labels = train_labels[:-1]

training_image = train_image/255
testing_image = test_image/255

X_train = training_image[0:50000,:]
X_test = training_image[50000:60000,:]
y_train = train_labels[0:50000,:]
y_test = train_labels[50000:60000,:]
#%%
X_t = testing_image
y_t = test_labels
#%%
y_train_ture = np.zeros([50000,10])
for j, class_idx in enumerate(y_train):
    y_train_ture[j,int(class_idx)] = 1

y_test_ture = np.zeros([10000,10])
for j, class_idx in enumerate(y_test):
    y_test_ture[j,int(class_idx)] = 1

y_t_ture = np.zeros([10000,10])
for j, class_idx in enumerate(y_t):
    y_t_ture[j,int(class_idx)] = 1
    
num_examples = X_train.shape[0]
epochs_completed = 0
index_in_epoch = 0
#%%
def shuffle(X, y):
    Z = np.column_stack((X, y))
    np.random.shuffle(Z)
    return Z[:, :-1], Z[:, -1]

def get_minibatch(X_train, y_train, minibatch_size,prob,session):
    X_train, y_train = shuffle(X_train, y_train)
    for i in range(0, X_train.shape[0], minibatch_size):
        # Get pair of (X, y) of the current minibatch/chunk
        X_train_mini = X_train[i:i + minibatch_size,:]
        y_train_mini = y_train[i:i + minibatch_size] 
        y_train_mini_ture = np.zeros([minibatch_size,10])
        for j, class_idx in enumerate(y_train_mini):
            y_train_mini_ture[j,int(class_idx)] = 1
        session.run(train,feed_dict={xs:X_train_mini,ys:y_train_mini_ture,keep_prob:prob}) 
        
def next_batch(batch_size):
    global X_train
    global y_train_ture
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all training data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        X_train = X_train[perm]
        y_train_ture = y_train_ture[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return X_train[start:end], y_train_ture[start:end]

def compute_accuracy(xs,ys,X,y,prob,session,prediction):
    y_pre = session.run(prediction,feed_dict={xs:X,keep_prob:prob})      
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) 
    result = session.run(accuracy,feed_dict={xs:X,ys:y,keep_prob:prob})
    return result 

#Weight Initialization
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  #mean=0 std=0.1
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)  # initialize bias as 0.1
    return tf.Variable(inital)

#Convolution and Pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')#strides[1]=1 strides[2]=1

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #strides[1]=2 strides[2]=2
#%%
xs = tf.placeholder(tf.float32,[None,784])  #the input data 24x24
ys = tf.placeholder(tf.float32,[None,10])  # the label of the input 
x_image = tf.reshape(xs,[-1,28,28,1]) # height=28, width=28, channel=1

#first convolution and max_pooling layer
W_conv1 = weight_variable([5,5,1,32]) # size of weights is 5x5, input channel is 1 and output channel is 6
b_conv1 = bias_variable([32])     # the number of the bias is 6     
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) # output is 28x28x6
h_pool1 = max_pool_2x2(h_conv1)     # output is 14x14x32

#second convolution and max_pooling layer
W_conv2 = weight_variable([5,5,32,64]) # size of weights is 5x5, input channel is 32 and output channel is 64
b_conv2 = bias_variable([64])   # the number of the bias is 64        
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)# output is 14x14x64
h_pool2 = max_pool_2x2(h_conv2)# output is 7x7x64

# first fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])# input layer is 3136
W_fc1 = weight_variable([7*7*64,1024])     # first hidden layer is 1024       
b_fc1 = bias_variable([1024])                    
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
# dropout
keep_prob = tf.placeholder(tf.float32)     # the dropout probablity
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# last fully connected layer
W_fc2 = weight_variable([1024, 10]) # output layer is 10
b_fc2 = bias_variable([10])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#softmax
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=ys))
#backpropagation
#train = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
train = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 调用梯度下降
init = tf.global_variables_initializer()
#%%
with tf.Session() as session:
    session.run(init)
    for i in range(10):
        #get_minibatch(X_train,y_train,1,0.5,session)
        batch_xs, batch_ys = next_batch(10)  
        session.run(train, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.5})
        print(session.run(loss,feed_dict={xs:X_train,ys:y_train_ture,keep_prob:0.5}))
        print(compute_accuracy(xs,ys,X_train,y_train_ture,0.5,session,prediction)) 
      
    

