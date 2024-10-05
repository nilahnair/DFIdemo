import torch
import torch.nn as nn 

class conditionalMoCapDiscriminator(nn.Module):

    def __init__(self, config):

        super(conditionalMoCapDiscriminator, self).__init__()

        # Same as MotionSense Autoencoder + 2 FCN layers 
        self.config = config

        self.stride_dim = self.config['stride'] 
        self.bias = self.config['bias']
        self.channels = self.config['channels']
        self.kernel = self.config['kernel']
        self.num_classes = self.config['num_classes']

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], self.kernel, self.stride_dim, self.config['padding_input'], padding_mode='replicate',bias=self.bias),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            
            nn.Conv2d(self.channels[1], self.channels[2], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(self.channels[2], self.channels[3], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )


        self.fc = nn.Sequential(
            #                                Depending on conv settings, need to update for different kernels
            nn.Linear(1 * self.channels[3] * 24 * 9 + self.num_classes, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)


    def forward(self, x, y):
        x = self.conv(x)
        x = x.reshape(-1, x.size()[1] * x.size()[2] * x.size()[3])
        
        y_embedded = self.label_embedding(y).view(x.size(0), -1)
        x = torch.cat((x, y_embedded), dim=1)
    
        return self.fc(x)