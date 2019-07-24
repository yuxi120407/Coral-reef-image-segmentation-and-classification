# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 09:25:15 2018

@author: yuxi
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import time
from torchsummary import summary
import numpy as np
from skimage.io import imread
import glob


#%%
def generate_signal_imagedata(path_txt,path_image,image_x,image_y):
    txt_file = open(path_txt)
    text = txt_file.readlines()[2:]
    count_points = len(text)
    
    crop_length = 30
    crop_width = 30
    all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
    label = np.zeros(count_points,dtype=np.int)
    crop_x = int(crop_length/2)
    crop_y = int(crop_width/2)
    image = imread(path_image)
    for i in range(count_points):
        text_piece = text[i]
        text_element = text_piece.split(',')
        l_x = int(float(text_element[0]))
        l_y = int(float(text_element[1]))
        label[i] = int(text_element[2])
        if(l_x-crop_x <0):
            l_x = crop_x
        if(l_y-crop_y <0):
            l_y = crop_y
        if(l_x+crop_x >image_y):
            l_x = image_y-15
        if(l_y+crop_y >image_x):
            l_y =image_x-15
        all_image[i,:,:,:] = image[l_y-15:l_y+15,l_x-15:l_x+15]
    txt_file.close()
    return all_image,label

def read_generate_data(txtfolder_name,imagefloder_name,image_x,image_y):
    all_count = 1
    all_label = np.zeros(1,dtype=np.int)
    image_data = np.zeros([1,30,30,3],dtype=np.uint8)
    read_files = glob.glob(str('./')+txtfolder_name+str('/*.txt'))
    for name in read_files:
        name = name.split("\\")[1]
        name = name.split(".")[0]
        path_txt = str('./')+txtfolder_name+str('/')+name+str('.txt')
        path_image = str('./')+ imagefloder_name+str('/')+name+str('.jpg')
        new_image_data,label = generate_signal_imagedata(path_txt,path_image,image_x,image_y)
        all_label = np.hstack((all_label,label))
        image_data = np.vstack((image_data,new_image_data))
        print("Image {0} is finish ".format(all_count))
        all_count = all_count+1
    final_data = image_data[1:,:,:,:]
    final_label = all_label[1:]
    return final_data,final_label

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size = (3, 3), 
                               stride = (1, 1), padding = (1, 1))
        self.conv1_no_padding = nn.Conv2d(32, 32,kernel_size = (3, 3),
                                          stride = (1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size = (3, 3),
                               stride = (1, 1), padding = (1, 1))
        self.conv2_no_padding = nn.Conv2d(64, 64, kernel_size = (3, 3),
                                          stride = (1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (3, 3),
                               stride = (1, 1), padding = (1, 1))
        self.conv3_no_padding = nn.Conv2d(128, 128, kernel_size = (3, 3),
                                          stride = (1, 1))
        
        
        self.fc1 = nn.Linear(2*2*128, 512)
        self.fc2 = nn.Linear(512, 6)
        self.relu = nn.ReLU()
        self.max_pooling = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv1_no_padding(x))
        x = self.max_pooling(x)
        x = self.dropout(x)
        
        
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2_no_padding(x))
        x = self.max_pooling(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv3_no_padding(x))
        x = self.max_pooling(x)
        x = self.dropout(x)
        
        
        x = x.view(-1, 2*2*128)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
#%%
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_time = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        start_time = time.time()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        epoch_time = epoch_time + end_time - start_time
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), epoch_time))
            epoch_time = 0

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%%
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#%%
data_2012_imagefloder_name = '2012image'
data_2012_txtfolder_name = '2012data_label augmentation(m=300)'
data2012_data,data2012_label = read_generate_data(data_2012_txtfolder_name,
                                                  data_2012_imagefloder_name,1536,2048)
#%%
img_rows = 30
img_cols = 30
x_train_data = data2012_data[0:50000]/255
x_train_data = x_train_data.reshape(x_train_data.shape[0], 3, img_rows, img_cols)
y_train_data = data2012_label[0:50000]


x_test_data = data2012_data[50000:]/255
x_test_data = x_test_data.reshape(x_test_data.shape[0], 3, img_rows, img_cols)
y_test_data = data2012_label[50000:]


#%%
x_train = torch.from_numpy(x_train_data).type('torch.FloatTensor')
y_train = torch.from_numpy(y_train_data).type('torch.LongTensor')
x_test = torch.from_numpy(x_test_data).type('torch.FloatTensor')
y_test = torch.from_numpy(y_test_data).type('torch.LongTensor')
#%%

train_dataset = TensorDataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = args.batch_size,
                                           shuffle = True, **kwargs)

test_dataset = TensorDataset(x_test,y_test)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = args.batch_size,
                                          shuffle = True, **kwargs)


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader)
