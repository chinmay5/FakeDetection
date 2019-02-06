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
# Inception module
class InceptionLayer(nn.Module):

    def __init__(self, in_channels=3, a=1, b=4, c=4, d=1):
        super(InceptionLayer, self).__init__()
        self.basic_conv_a = nn.Conv2d(in_channels=in_channels, out_channels=a, kernel_size=1)
        self.bn_a = nn.BatchNorm2d(a, eps=0.001)
        # In all these cases, network passes through layer conv1X1 first
        self.basic_conv_b = nn.Conv2d(in_channels=a, out_channels=b, kernel_size=3, padding=1)
        self.bn_b = nn.BatchNorm2d(b, eps=0.001)
        self.basic_conv_c = nn.Conv2d(in_channels=a, out_channels=c, kernel_size=3, padding=2, dilation=2)
        self.bn_c = nn.BatchNorm2d(c, eps=0.001)
        self.basic_conv_d = nn.Conv2d(in_channels=a, out_channels=d, kernel_size=3, padding=3, dilation=3)
        self.bn_d = nn.BatchNorm2d(d, eps=0.001)
        
    def forward(self, x):
        branch1 = self.basic_conv_a(x)
        branch1 = F.relu(self.bn_a(branch1), inplace=True)
        
        branch3_d0 = self.basic_conv_a(x)
        branch3_d0 = self.basic_conv_b(branch3_d0)
        branch3_d0 = F.relu(self.bn_b(branch3_d0), inplace=True)

        branch3_d1 = self.basic_conv_a(x)
        branch3_d1 = self.basic_conv_c(branch3_d1)
        branch3_d1 = F.relu(self.bn_c(branch3_d1), inplace=True)
        
        branch3_d2 = self.basic_conv_a(x)
        branch3_d2 = self.basic_conv_d(branch3_d2)
        branch3_d2 = F.relu(self.bn_d(branch3_d2), inplace=True)
        
        outputs = [branch1, branch3_d0, branch3_d1, branch3_d2]
        return torch.cat(outputs, 1)

# Main file
class Meso_Incepption_v2(nn.Module):
    def __init__(self, in_channels=3, img_shape=256, number_of_classes=2):
        super(Meso_Incepption_v2,self).__init__()
        self.inception_layer_1 = InceptionLayer(in_channels, a=1, b=4, c=4, d=1)
        self.inception_layer_2 = InceptionLayer(in_channels=10, a=1, b=4, c=4, d=2) #First layer had 10 channels
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=(4, 4), stride=4)
        self.conv_5_1 = nn.Conv2d(in_channels=11, out_channels=16, kernel_size=5, padding=2)#Sum of channels
        self.conv_5_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2)#Sum of channels
        final_h_w = new_output_shape(num_of_maxpool_2=5, shape=img_shape)#We have a MaxPool of 4 included here
        self.fc_conv = nn.Conv2d(16,number_of_classes,kernel_size=(final_h_w,final_h_w))
        
    def forward(self,x_3d):
        cnn_embed_seq = []
        x_3d = x_3d.permute(0,1,4,2,3)# Required to match shapes
        x_3d = x_3d.type(torch.cuda.FloatTensor) #Converting to Float Tensor from Byte Tensor
        for t in range(x_3d.size(1)):
            x1 = self.inception_layer_1(x_3d[:, t, :, :, :])
            x1 = self.max_pool_2(x1)
            # Second layer
            x2 = self.inception_layer_2(x1)
            x2 = self.max_pool_2(x2)
            # Third layer
            x3 = self.conv_5_1(x2)
            x3 = F.relu(self.max_pool_2(x3), inplace=True)
            # Fourth layer
            x4 = self.conv_5_2(x3)
            x4 = F.relu(self.max_pool_4(x4), inplace=True)
            # Now flatten the layers
            x = self.fc_conv(x4)
            #print("Shape of x is {}".format(x.shape))
            x = x.view(x.shape[0], -1)
            
            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq
