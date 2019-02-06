import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable

def new_output_shape(num_of_maxpool_2,shape):
    return int(shape/(2**num_of_maxpool_2))
# module definition class
class Meso4(nn.Module):
    def __init__(self,in_channel=3, img_shape = 256, number_of_classes=2):
        super(Meso4,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,8, kernel_size=(3,3), stride = 1, padding= 1)
        self.batch_norm_1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8,8, kernel_size=(5,5),stride=1, padding=2)
        self.batch_norm_2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16, kernel_size=(5,5),stride=1, padding=2)
        self.batch_norm_3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,16, kernel_size=(5,5),stride=1, padding=2)
        self.batch_norm_4 = nn.BatchNorm2d(16)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        out_shape = new_output_shape(num_of_maxpool_2=4, shape = img_shape)
        self.fc_conv = nn.Conv2d(16,number_of_classes,kernel_size=(out_shape,out_shape))
        self.dropout = nn.Dropout2d(p=0.2)
        
    def forward(self,x_3d):
        cnn_embed_seq = []
        x_3d = x_3d.permute(0,1,4,2,3)# Required to match shapes
        x_3d = x_3d.type(torch.cuda.FloatTensor) #Converting to Float Tensor from Byte Tensor
        for t in range(x_3d.size(1)):
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.batch_norm_1(x)
            x = self.relu(x)
            
            # x = self.dropout(x) Avoid in the first layer
            # Segment 2
            x = self.max_pool_2(x)
            x = self.conv2(x)
            x = self.batch_norm_2(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            # Segment 3
            x = self.max_pool_2(x)
            x = self.conv3(x)
            x = self.batch_norm_3(x)
            x = self.relu(x)
            #x = self.dropout(x)
            
            # Segment 4
            x = self.max_pool_2(x)
            x = self.conv4(x)
            x = self.batch_norm_4(x)
            x = self.relu(x)
            x = self.dropout(x)
            
            # Going for the last layer
            x = self.max_pool_2(x)
            x = self.fc_conv(x)
            #print("Shape of x is {}".format(x.shape))
            x = x.view(x.shape[0], -1)
            
            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq