import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Encoder(nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan

        self.encoder = nn.Sequential(
            # First Conv Block: im_chan -> hidden_dim
            nn.Conv2d(im_chan, hidden_dim, kernel_size=3, stride=2, padding=1),  # Output: 32x32
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Second Conv Block: hidden_dim -> hidden_dim * 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # Output: 16x16
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # Third Conv Block: hidden_dim * 2 -> output_chan * 2
            nn.Conv2d(hidden_dim * 2, output_chan * 2, kernel_size=3, stride=2, padding=1),  # Output: 8x8
            nn.BatchNorm2d(output_chan * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_chan * 2, output_chan * 2, kernel_size=3, stride=2, padding=0)  # Output: 4x1x1
        )

    def forward(self, image):
        encoder_pred = self.encoder(image)
        encoding = encoder_pred.view(len(encoder_pred), -1)
        mean = encoding[:, :self.z_dim]
        logvar = encoding[:, self.z_dim:]

        return mean, torch.exp(logvar * 0.5)


class Decoder(nn.Module):
    def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.decoder = nn.Sequential(
            self.init_conv_block(z_dim, hidden_dim * 4),
            self.init_conv_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.init_conv_block(hidden_dim * 2, hidden_dim),
            self.init_conv_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def init_conv_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        else:
            layers += [nn.Sigmoid()]
        return nn.Sequential(*layers)

    def forward(self, z):
        x = z.view(-1, self.z_dim, 1, 1)
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, z_dim=32, im_chan=1):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(im_chan, z_dim)
        self.decoder = Decoder(z_dim, im_chan)

    def forward(self, images):
        mean, std = self.encoder(images)
        z_dist = Normal(mean, std)
        z = z_dist.rsample()
        decoding = self.decoder(z)
        return decoding, z_dist