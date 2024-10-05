import torch
import torch.nn as nn 
import torch.nn.functional as F

class MoCapGenerator(nn.Module):

    filter_dim = 4
    filter_dim_deconv = filter_dim
    stride_dim = 2
    padding_dim = (5, 0)
    
    def __init__(self, config):
        super(MoCapGenerator, self).__init__()
        
        self.config = config

        self.stride_dim = self.config['stride'] 
        self.bias = self.config['bias']
        self.channels = self.config['channels']
        self.kernel = self.config['kernel']
        self.num_classes = self.config['num_classes']
        self.disable_embedding = self.config['disable_embedding']
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[1], self.kernel, self.stride_dim, self.config['padding_input'], padding_mode='replicate',bias=self.bias),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            
            nn.Conv2d(self.channels[1], self.channels[2], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(self.channels[2], self.channels[3], self.kernel, self.stride_dim, 0, bias=self.bias),
            # Toggle for MS45
            #nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        
        # Bottleneck (combination of latent and encoded features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.channels[3], self.channels[3], 1, 1, 0, bias=self.bias),
            nn.ReLU(),        
        )
        
        if self.disable_embedding: 
            self.rv_dense = nn.Sequential(
                nn.Linear(32, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),  # Use LeakyReLU with a small negative slope

                nn.Linear(1 * self.channels[3] * 24 * 9, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),

                nn.Linear(1 * self.channels[3] * 24 * 9, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),
            )
        else: 
            self.rv_dense = nn.Sequential(
                nn.Linear(32 + self.num_classes, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),  # Use LeakyReLU with a small negative slope

                nn.Linear(1 * self.channels[3] * 24 * 9, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),

                nn.Linear(1 * self.channels[3] * 24 * 9, 1 * self.channels[3] * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),
            )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.channels[3], self.channels[4], self.kernel, self.stride_dim, 0, bias=self.bias),
            # Toggle for MS45
            #nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(self.channels[4], self.channels[5], self.kernel, self.stride_dim, 0, bias=self.bias),
            nn.BatchNorm2d(self.channels[5]),
            nn.ReLU(),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(self.channels[5], self.channels[6], self.kernel, self.stride_dim, self.config['padding_output'], bias=self.bias),      
            nn.Sigmoid()  
        )
        
        self.label_embedding = nn.Embedding(self.num_classes, self.num_classes)
        
        
    def load_pretrained_ae(self, path):
        # Load the pretrained encoder, decoder and bottleneck
        # does NOT load any rv layers 
        
        model = torch.load(path)
        self_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in model.state_dict().items() if k in self_dict}
        self_dict.update(pretrained_dict) 
        self.load_state_dict(self_dict)
        
    def decode(self, x):
        dec = self.decoder(x)    
        return dec
    
    def encode_rv(self, latent_vector, y=None):
        if y is not None:
            y_embedded = self.label_embedding(y).view(latent_vector.size(0), 1, 1, -1)
            combined_input = torch.cat((latent_vector, y_embedded), dim=3)
        else: 
            combined_input = latent_vector
        
        rvc = self.rv_dense(combined_input)
        rvc = rvc.view(latent_vector.size(0), -1, 24, 9)
        return rvc
    
    def encode(self, x):
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        return bottle
    
    def dfi(self, x_enc, latent_vector, y):
        bottle = x_enc
        
        # Calculate feature map for latent vector input
        rvc = self.encode_rv(latent_vector, y)
        
        # apply a DFI for the encoded vectors
        combined = bottle + rvc
            
        # Decode
        dec = self.decoder(combined)               
        return dec[:,:,0:200,:]
        

    def forward(self, x, latent_vector, y=None):
        # Encode the baseline image
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        # Calculate feature map for latent vector input
        rvc = self.encode_rv(latent_vector, y)
        
        # apply a DFI for the encoded vectors
        combined = bottle + rvc
            
        # Decode
        dec = self.decoder(combined)               
        return dec[:,:,0:200,:]