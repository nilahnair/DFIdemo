import torch.nn as nn 

class MotionSenseDiscriminator(nn.Module):

    def __init__(self, config):

        super(MotionSenseDiscriminator, self).__init__()

        # Same as MotionSense Autoencoder + 2 FCN layers 
        self.config = config

        self.stride_dim = self.config['stride'] 
        self.bias = self.config['bias']
        self.channels = self.config['channels']
        self.kernel = self.config['kernel']

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
            nn.Linear(1 * self.channels[3] * 24 * 9, 128),
            nn.ReLU(), 
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, x.size()[1] * x.size()[2] * x.size()[3])
    
        return self.fc(x)