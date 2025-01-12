# load packages 
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


# construct encoder 
class Encoder(nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan
        
        # define a sequence of convolutional, batch normalization, and activation layers
        self.encoder = nn.Sequential(
            nn.Conv2d(im_chan, hidden_dim, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_dim * 2, output_chan * 2, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(output_chan * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_chan * 2, output_chan * 2, kernel_size=3, stride=2, padding=0)  
        )

    def forward(self, image):
        # pass the input through the encoder 
        encoder_pred = self.encoder(image)
        # flatten the output and split it into mean and log variance 
        encoding = encoder_pred.view(len(encoder_pred), -1)
        mean = encoding[:, :self.z_dim]
        logvar = encoding[:, self.z_dim:]

        # return mean and standard deviation 
        return mean, torch.exp(logvar * 0.5)


# construct decoder 
class Decoder(nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim

        # define a sequence of transposed convolution, batch normalization and activation layers
        # approximately mirrow to the encoder 
        # to achieve size 28x28 in the end, a even kernel size of 4 could not been avoided  
        self.decoder = nn.Sequential(
             nn.ConvTranspose2d(z_dim, hidden_dim * 4, kernel_size=3, stride=2, padding=0),  
             nn.BatchNorm2d(hidden_dim * 4),
             nn.ReLU(inplace=True),

             nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1, padding=0),  
             nn.BatchNorm2d(hidden_dim * 2),
             nn.ReLU(inplace=True), 

             nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=0),  
             nn.BatchNorm2d(hidden_dim),
             nn.ReLU(inplace=True), 
 
             nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=0),
             nn.Sigmoid()
             )

    # reshape the latent vector and pass through the encoder 
    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        return self.decoder(x)

# construct Variational Autoencoder (VAE) 
class VAE(nn.Module):
    def __init__(self, z_dim=32, im_chan=1):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        # instantiate an object of the Encoder class
        self.encoder = Encoder(im_chan, z_dim)
        # instantiate an object of the decoder class
        self.decoder = Decoder(z_dim, im_chan)

    def forward(self, images):
        # call encoder to map image to latent space 
        mean, std = self.encoder(images)
        # define a normal distributed variable with mean mean and std deviation std 
        z_dist = Normal(mean, std)
        z = z_dist.rsample()
        # call decoder to map latent vector to reconstructed image 
        decoding = self.decoder(z)
        return decoding, z_dist