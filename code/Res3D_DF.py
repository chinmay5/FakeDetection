import torch.nn as nn
import torch


def new_output_shape(num_of_maxpool_2,shape):
    if shape==256:
        return 9
    elif shape==128:
        return 5
    return -1

def new_channel_shape(frames):
    if frames==10:
        return 1
    # for 30 frames, the shape is 
    return 2

class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, img_dim, frames,dropout):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        
        # Let us see what removing conv_3 does
        
        self.conv3a = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(64)
        
        self.conv4a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn4 = nn.BatchNorm3d(128)
        
        self.conv5a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Logic for using 1X1 conv to reduce params
        dim_shape = new_output_shape(num_of_maxpool_2=5, shape=img_dim)
        channel_shape = new_channel_shape(frames=frames) # As first time channel is not modified
        # print("channel shape is {} and dimension shape is {}".format(channel_shape, dim_shape)) 
        self.fc_conv = nn.Conv3d(256, 2, kernel_size=(channel_shape, dim_shape, dim_shape))
        
        self.dropout = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor) # Shape of X is torch.Size([4, 30, 3, 128, 128])
        x = x.permute(0,2,1,3,4)

        # Permutation done to move the channels to beginning which simplifies computation
        h = self.relu(self.conv1(x))
        h = self.bn1(h) # include batch norm after relu where it seems to perform better
        h = self.pool1(h)
        h = self.dropout(h)
        #print("shape layer 1 is {}".format(h.shape))
        
        h = self.relu(self.conv2(h))
        h = self.bn2(h) #include batch norm
        h = self.pool2(h)
        h = self.dropout(h)
        #print("shape is layer 2 is {}".format(h.shape))
        
        # let us see effect of removing conv_3
        
#         h = self.relu(self.conv3a(h))
#         h = self.relu(self.conv3b(h))
#         h = self.bn3(h) #batch norm layer
#         h = self.pool3(h)
#         h = self.dropout(h)
        #print("shape is layer 3{}".format(h.shape))
        
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.bn4(h) #including batch norm layer 
        h = self.pool4(h)
        h = self.dropout(h)
        #print("shape is layer 4 {}".format(h.shape))
        
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        #print("shape is at end {}".format(h.shape))
        h = self.fc_conv(h)
        #h = h.view(-1, 8192)
        #h = self.relu(self.fc6(h))
        #h = self.dropout(h)
        #h = self.relu(self.fc7(h))
        #h = self.dropout(h)
        
        logits = h.view(x.shape[0],-1)
        
        #logits = self.fc8(h)
        # we return logits and handle rest using BCE loss
        # probs = self.softmax(logits)
        #print("shape returned is {}".format(logits.shape))
        return logits

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""
