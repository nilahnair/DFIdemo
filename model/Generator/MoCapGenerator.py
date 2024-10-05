import torch
import torch.nn as nn 
import torch.nn.functional as F

class MoCapGenerator(nn.Module):

    filter_dim = 4
    filter_dim_deconv = filter_dim
    stride_dim = 2
    padding_dim = (5, 0)
    num_classes = 16
    
    def __init__(self, config):
        super(MoCapGenerator, self).__init__()
        
        # Encoder (baseline image)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, self.filter_dim, self.stride_dim, self.padding_dim, padding_mode='replicate',bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, self.filter_dim, self.stride_dim, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Conv2d(32, 64, self.filter_dim, self.stride_dim, 0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Bottleneck (combination of latent and encoded features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.ReLU(),        
        )

        self.rv_dense = nn.Sequential(
                nn.Linear(32 + self.num_classes, 1 * 64 * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),  # Use LeakyReLU with a small negative slope

                nn.Linear(1 * 64 * 24 * 9, 1 * 64 * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),

                nn.Linear(1 * 64 * 24 * 9, 1 * 64 * 24 * 9, bias=True),
                nn.LeakyReLU(0.2),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, self.filter_dim_deconv, self.stride_dim, 0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),        
            nn.Dropout(p=0.1),

            nn.ConvTranspose2d(32, 16, self.filter_dim_deconv, self.stride_dim, 0, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(),        
            nn.Dropout(p=0.1),

            nn.ConvTranspose2d(16, 1, self.filter_dim_deconv, self.stride_dim, (3, 0), bias=True),      
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
        rvc = rvc.view(latent_vector.size(0), -1, 24, 64)
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