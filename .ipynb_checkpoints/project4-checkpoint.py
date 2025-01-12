import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import time
import numpy as np
import os

# Define the reconstruction loss
reconstruction_loss = nn.BCELoss(reduction='sum')

def kl_divergence_loss(z_dist):
    return kl_divergence(z_dist,
                         Normal(torch.zeros_like(z_dist.mean),
                                torch.ones_like(z_dist.stddev))
                         ).sum(-1).sum()

class Encoder(nn.Module):
    def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
        super(Encoder, self).__init__()
        self.z_dim = output_chan

        self.encoder = nn.Sequential(
            self.init_conv_block(im_chan, hidden_dim),
            self.init_conv_block(hidden_dim, hidden_dim * 2),
            self.init_conv_block(hidden_dim * 2, output_chan * 2, final_layer=True),
        )

    def init_conv_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
        layers = [
            nn.Conv2d(input_channels, output_channels,
                          kernel_size=kernel_size,
                          padding=padding,
                          stride=stride)
        ]
        if not final_layer:
            layers += [
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers)

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

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('.', download=True, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

def show_images_grid(images, title='Sample Images Grid', save=False):
    '''
    show input torch tensor of images [num_images, ch, w, h] in a grid
    '''
    plt.figure(figsize=(7, 7))
    grid = vutils.make_grid(images, nrow=images.shape[0]//2, padding=2, normalize=True)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    if save:
        date = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(date + '/' + title + '.png')
    else:
        plt.show()
    # close the plot
    plt.close()

def train_model(epochs=10, z_dim=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(z_dim=z_dim).to(device)
    model_opt = torch.optim.Adam(model.parameters())

    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        for images, _ in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            model_opt.zero_grad()
            recon_images, encoding = model(images)
            loss = reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding)
            loss.backward()
            model_opt.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        training_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Validation"):
                images = images.to(device)
                recon_images, encoding = model(images)
                loss = reconstruction_loss(recon_images, images) + kl_divergence_loss(encoding)
                val_loss += loss.item()

        val_loss /= len(test_loader.dataset)
        validation_losses.append(val_loss)

        print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Visualization placeholder
        show_images_grid(images.cpu(), title=f'Input images after epoch {epoch + 1}', save=True)
        show_images_grid(recon_images.cpu(), title=f'Reconstructed images after epoch {epoch + 1}', save=True)

    return model, training_losses, validation_losses

if __name__ == "__main__":
    z_dim = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # date and time for model name

    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = 'vae_' + str(z_dim) + '_' + date

    # create folder with date and time
    os.makedirs(date, exist_ok=True)


    # if model exists, load it else train it
    try:
        vae = VAE(z_dim=z_dim)
        vae.load_state_dict(torch.load(model_name + '.pth'))
    except FileNotFoundError:
        vae, train_losses, val_losses = train_model(epochs=5, z_dim=z_dim)
        torch.save(vae.state_dict(), date + '/' + model_name)

        # save training and validation losses with numpy .npy
        np.save(date + '/' + 'train_losses_' + model_name + '.npy', np.array(train_losses))
        np.save(date + '/' + 'val_losses_' + model_name + '.npy', np.array(val_losses))


    # show some test images
    for images, _ in test_loader:
        images = images.to(device)
        recon_images, encoding = vae(images)
        show_images_grid(images.cpu(), title=f'Input images')
        show_images_grid(recon_images.cpu(), title=f'Reconstructed images')
        break