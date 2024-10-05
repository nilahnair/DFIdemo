import torch
import torch.nn.functional as F
import torch.nn as nn

class DeepConvolutionalAutoencoder(nn.Module):
        
   
    def __init__(self, config):
        super(DeepConvolutionalAutoencoder, self).__init__()

        self.config = config

        self.stride_dim = self.config['stride'] 
        self.bias = self.config['bias']
        self.channels = self.config['channels']
        self.kernel = self.config['kernel']
        
        # Encoder (baseline image)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], self.kernel, self.stride_dim, self.config['padding_input'], padding_mode='replicate', bias=self.bias),
            nn.BatchNorm2d(self.channels[1]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(self.channels[1], self.channels[2], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(self.channels[2], self.channels[3], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),

            nn.Conv2d(self.channels[3], self.channels[4], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[4]),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(self.channels[4], self.channels[5], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[5]),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.1),
        )
        
        # Bottleneck (combination of latent and encoded features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channels[5], self.channels[3], 1, 1, 0, bias=self.bias),
            nn.LeakyReLU(0.2),        
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channels[5], self.channels[6], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[6]),
            nn.LeakyReLU(0.2),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(self.channels[6], self.channels[7], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[7]),
            nn.LeakyReLU(0.2),        
            nn.Dropout(p=0.1),

            nn.ConvTranspose2d(self.channels[7], self.channels[8], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[8]),
            nn.LeakyReLU(0.2),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(self.channels[8], self.channels[9], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[9]),
            nn.LeakyReLU(0.2),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(self.channels[9], self.channels[10], self.kernel, self.stride_dim, self.config['padding_output'], bias=self.bias),      
            nn.Sigmoid()  
        )
        
    def decode(self, x):
        dec = self.decoder(x)    
        return dec
    
    def encode(self, x):
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        return bottle

    def forward(self, x):
        # Encode the baseline image
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        # Decode
        dec = self.decoder(bottle)               
        return dec