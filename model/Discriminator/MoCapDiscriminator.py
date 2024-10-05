import os
import pickle

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import pandas as pd
import numpy as np 

class ImuLaraDiscriminator(nn.Module):
    """A Standalone Discriminator for OMoCap data, as given by the main network.

    The idea is to re-use convolution filters as trained by the other network.
    The structure is similar up to the first MLP layer, so conv_1_1, ..., conv_2_2 remain
    untouched but the three MLP layers are re-trained to avoid implicit bias while keeping
    the pre-trained filters or at least use them to bootstrap the process.

    Structure:
        ImuLaraDiscriminator(
          (conv1_1): Conv2d(1, 64, kernel_size=(5, 1), stride=(1, 1))
          (conv1_2): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
          (conv2_1): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
          (conv2_2): Conv2d(64, 64, kernel_size=(5, 1), stride=(1, 1))
          (fc3): Linear(in_features=1580544, out_features=256, bias=True)
          (fc4): Linear(in_features=256, out_features=256, bias=True)
          (fc_final): Linear(in_features=256, out_features=1, bias=True)
          (avgpool): AvgPool2d(kernel_size=[1, 126], stride=[1, 126], padding=0)
          (sigmoid): Sigmoid()
        )
    """

    def __init__(self, config):
        '''
        Constructor
        '''

        super(ImuLaraDiscriminator, self).__init__()


        self.config = config

        Hx = 126 #self.config['NB_sensor_channels']
        Wx = 200 #self.config['sliding_window_length']
        in_channels = 1 
        filter_dim = 4
        padding_dim = (5, 0)
        num_filter=32
        stride_dim = 2
        self.num_classes = 16


        # Computing the size of the feature maps
     
        '''
        Wx, Hx = self.size_feature_map(Wx=Wx,
                                       Hx=Hx,
                                       F=(4, 1),
                                       P=(5, 0), S=(1, 1), type_layer='conv')
        '''
        # set the Conv layers

        self.discriminate = nn.Sequential(
        nn.Conv2d(1,2, filter_dim, stride_dim, padding_dim, padding_mode='replicate',bias=True),
        nn.BatchNorm2d(2),
        nn.ReLU(),

        nn.Conv2d(2, 4, filter_dim, stride_dim, 0, bias=True),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Dropout(p=0.1),

        nn.Conv2d(4, 4, filter_dim, stride_dim, 0, bias=True),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        )
        # MLP
        #self.fc3 = nn.Linear(126, 126)
        self.fc3 = nn.Linear(34* 4 *10, 256) #num_filter * 184 * 126, 256

        #self.fc3_RL = nn.Linear(num_filter * int(Wx) * 24, 256)

        self.fc4 = nn.Linear(256, 126)
        self.fc_final = nn.Linear(126, 1) #only 1 class, fake or real
        self.avgpool = nn.AvgPool2d(kernel_size=[1, 126])

        self.sigmoid = nn.Sigmoid()

        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)

        return


    def forward(self, x, y):
        '''
        Forwards function, required by torch.

        @param x: batch [batch, 1, Channels, Time], Channels = Sensors * 3 Axis
        @return x: Output of the network, either Softmax or Attribute
        '''

        x = self.discriminate(x)
        # x = F.max_pool2d(x, (2, 1))
        #### QUESTION: Why is there no pooling for FC network?
        ### Fernando's thesis has shown that pooling does not help here, at least no spatial pooling
        # view is reshape
        # x.view(x.size(0), -1)
        print('descriminator block')
        print(x.shape)
        x = x.reshape(-1, x.size()[1] * x.size()[2] * x.size()[3])
        
        y_embedded = self.label_embedding(y).view(x.size(0), -1)
        x = torch.cat((x, y_embedded), dim=1)
        print('after embedding')
        print(x.shape)
        
        # x = x.permute(0, 2, 1)
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc3_RL(x))


        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))


        x = self.sigmoid(self.fc_final(x))
        print('final x')
        print(x.shape)
        return x

    def size_feature_map(self, Wx, Hx, F, P, S, type_layer = 'conv'):
        '''
        Computing size of feature map after convolution or pooling

        @param Wx: Width input
        @param Hx: Height input
        @param F: Filter size
        @param P: Padding
        @param S: Stride
        @param type_layer: conv or pool
        @return Wy: Width output
        @return Hy: Height output
        '''

        Pw = P
        Ph = P

        if type_layer == 'conv':
            Wy = 1 + (Wx - F[0] + 2 * Pw) / S[0]
            Hy = 1 + (Hx - F[1] + 2 * Ph) / S[1]

        elif type_layer == 'pool':
            Wy = 1 + (Wx - F[0]) / S[0]
            Hy = 1 + (Hx - F[1]) / S[1]

        return Wy, Hy