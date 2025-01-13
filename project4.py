# load packages 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torchvision.utils as 
import tensorflow_probability as tfp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# load Encoder, Decoder, and VAE 
from model import Encoder, Decoder, VAE

# class Encoder(nn.Module):
#     def __init__(self, im_chan=1, output_chan=32, hidden_dim=16):
#         super(Encoder, self).__init__()
#         self.z_dim = output_chan

#         self.encoder = nn.Sequential(
#             # First Conv Block: im_chan -> hidden_dim
#             nn.Conv2d(im_chan, hidden_dim, kernel_size=3, stride=2, padding=1),  # Output: 32x32
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
            
#             # Second Conv Block: hidden_dim -> hidden_dim * 2
#             nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # Output: 16x16
#             nn.BatchNorm2d(hidden_dim * 2),
#             nn.ReLU(inplace=True),
            
#             # Third Conv Block: hidden_dim * 2 -> output_chan * 2
#             nn.Conv2d(hidden_dim * 2, output_chan * 2, kernel_size=3, stride=2, padding=1),  # Output: 8x8
#             nn.BatchNorm2d(output_chan * 2),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(output_chan * 2, output_chan * 2, kernel_size=3, stride=2, padding=0)  # Output: 4x1x1
#             )

#     def forward(self, image):
#         encoder_pred = self.encoder(image)
#         encoding = encoder_pred.view(len(encoder_pred), -1)
#         mean = encoding[:, :self.z_dim]
#         logvar = encoding[:, self.z_dim:]

#         return mean, torch.exp(logvar * 0.5)


# class Decoder(nn.Module):
#     def __init__(self, z_dim=32, im_chan=1, hidden_dim=64):
#         super(Decoder, self).__init__()
#         self.z_dim = z_dim

#         self.decoder = nn.Sequential(
#             # First Transposed Conv Block: z_dim -> hidden_dim * 4
#             nn.ConvTranspose2d(z_dim, hidden_dim * 4, kernel_size=3, stride=2, padding=0),  
#             nn.BatchNorm2d(hidden_dim * 4),
#             nn.ReLU(inplace=True),

#             # Second Transposed Conv Block: hidden_dim * 4 -> hidden_dim * 2
#             nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1, padding=0),  
#             nn.BatchNorm2d(hidden_dim * 2),
#             nn.ReLU(inplace=True), 

#             # Third Transposed Conv Block: hidden_dim * 2 -> hidden_dim
#             nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=0),  
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True), 
 
#             # Final Transposed Conv Block: hidden_dim -> im_chan (final output: 28x28)
#             nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=0),
#             nn.Sigmoid()
#             )

#     def init_conv_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
#         layers = [
#             nn.ConvTranspose2d(input_channels, output_channels,
#                                kernel_size=kernel_size,
#                                stride=stride, padding=padding)
#         ]
#         if not final_layer:
#             layers += [
#                 nn.BatchNorm2d(output_channels),
#                 nn.ReLU(inplace=True)
#             ]
#         else:
#             layers += [nn.Sigmoid()]
#         return nn.Sequential(*layers)

#     def forward(self, z):
#         x = z.view(-1, self.z_dim, 1, 1)
#         return self.decoder(x)

# class VAE(nn.Module):
#     def __init__(self, z_dim=32, im_chan=1):
#         super(VAE, self).__init__()
#         self.z_dim = z_dim
#         self.encoder = Encoder(im_chan, z_dim)
#         self.decoder = Decoder(z_dim, im_chan)

#     def forward(self, images):
#         mean, std = self.encoder(images)
#         z_dist = Normal(mean, std)
#         z = z_dist.rsample()
#         decoding = self.decoder(z)
#         return decoding, z_dist

# DATA PREPARATION 
# define data transformation 
transform = transforms.Compose([
    transforms.ToTensor(),                          # Convert image to tensor
    transforms.Lambda(lambda x: (x > 0.5).float())  # Threshold: convert pixels above 0.5 to 1, others to 0
])

# load MNIST dataset, apply data transformation and create DataLoader  
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST('.', download=True, train=False, transform=transform)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)


# VISUALIZATION METHODS 
# Function to visualize images in a grid 
def show_images_grid(images, date, title='Sample Images Grid', save=False):
    '''
    Display a grid of images from a PyTorch tensor.
    Parameters:
    - images: tensor of shape [num_images, channels, height, width]
    - date: to differ between different time, used in file name  
    - title: title for the plot
    - save: save the plot to file if True
    '''
    
    plt.figure(figsize=(7, 7))
    # create a grid of images 
    grid = vutils.make_grid(images, nrow=images.shape[0]//2, padding=2, normalize=True)
    # rearrange dimensions for display 
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    # save or display the plot 
    if save:
        plt.savefig(date + '/' + title + '.png')
    else:
        plt.show()
    # close the plot
    plt.close()

# function to visualize the latent space 
def plot_latent_images(model, n, epoch, z_dim=2, im_size=28, save=True, date=None, frame=None):
    """
    Generate a grid of images by sampling from the latent space.

    Parameters:
    - model: the trained VAE model.
    - n: the number of images per dimension in the grid.
    - epoch: current training epoch.
    - z_dim: dimensionality of the latent space.
    - im_size: size of each generated image.
    - save: whether to save the resulting image grid.
    - save_dir: directory to save the images if save=True.
    """

    # ensure the model is in evaluation mode
    model.eval()

    # initialize the dimensions of the ouput image grid 
    image_width = im_size * n # width of grid 
    image_height = image_width # height of grid 
    image = np.zeros((image_height, image_width)) # empty numpy array to store final grid 

    # generate evenly spaced points in the latent space
    grid_values = torch.linspace(-2, 2, steps=n)

    # generate images for each point in the grid
    # disable gradient calculation for efficiency 
    with torch.no_grad(): 
        # loop through vertical and horizontal grid positions 
        for i, yi in enumerate(grid_values):
            for j, xi in enumerate(grid_values):
                # create a latent vector for the current grid coordinates 
                z = torch.tensor([[xi, yi]], dtype=torch.float32).to(next(model.parameters()).device)
                # adjust dimensions to match decoder's input requirements 
                z = z.view(-1, z_dim)  
                # decode the latent vector to generate an image 
                decoded_image = model.decoder(z)
                # process the decoded image to remove extra dimensions 
                decoded_image = decoded_image.squeeze(0).squeeze(0).cpu().numpy()  # (1, 1, 28, 28) -> (28, 28)
                # place the decoded image in the grid 
                image[i * im_size: (i + 1) * im_size,
                      j * im_size: (j + 1) * im_size] = decoded_image

    # plot the resulting image grid
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('off')
    plt.title(f"Latent Space Grid at Epoch {epoch}")

    # save or show the grid if required
    if save:
        if epoch == 0:
            plt.savefig(f"{date}/images/latent_grid_epoch_{epoch:03d}.{frame:03d}.png")
        else:
            plt.savefig(f"{date}/images/latent_grid_epoch_{epoch:03d}.png")
    else:
        plt.show()

    # close the plot
    plt.close()


# function to plot the training and validation losses 
def plot_losses(val_loss, train_loss, date=None, title='Training and Validation Losses'):
    """
    Plot the training and validation loss 

    Parameters:
    - val_loss: validation loss 
    - train_loss: train_loss 
    - date: to uniquely name the plot provide a date 
    - title: title for the plot  
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if date:
        plt.savefig(date + '.png')
    
    plt.show()


# DEFINE LOSS FUNCTION 
# define the reconstruction loss by Binary Cross-Entropy (sum over all pixels)
reconstruction_loss = nn.BCELoss(reduction='sum')

# define KL-divergence 
def kl_divergence_loss(z_dist):
    # calculate KL-divergence between the latent distribution (z_dist)
    # and a standard normal distribution
    return kl_divergence(z_dist,
                         Normal(torch.zeros_like(z_dist.mean),
                                torch.ones_like(z_dist.stddev))
                         ).sum(-1).sum()

# TRAINING PROCESS 
# Training function for the VAE model 
def train_model(date, epochs=10, z_dim=16):
    # use GPU if available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize VAE model 
    model = VAE(z_dim=z_dim).to(device)
    # use Adam optimizer 
    model_opt = torch.optim.Adam(model.parameters())

    # initialize lists to store training and validation losses 
    training_losses = []
    validation_losses = []

    training_recon_losses = []
    validation_recon_losses = []
    
    training_kl_losses = []
    validation_kl_losses = []

    # loop over epochs 
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        # initialize idx for plotting the latent space 
        idx = 0
        # loop over training batches 
        for images, _ in tqdm(train_loader, desc="Training"):
            # move images to the appropriate device 
            images = images.to(device)
            # reset gradients 
            model_opt.zero_grad()

            # forward pass through the VAE 
            recon_images, encoding = model(images)
            # compute reconstruction loss. KL-divergence loss and sum to ELBO loss  
            recon_loss = reconstruction_loss(recon_images, images)
            kl_loss = kl_divergence_loss(encoding)
            loss = recon_loss + kl_loss
            # backpropagation 
            loss.backward()
            # update model weights 
            model_opt.step()

            # accumulate losses 
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

            # in the first epoch and for latent dimension 2 plot the latent space for every 75 update steps 
            if epoch == 0 and idx % 75 == 0 and z_dim == 2:
                plot_latent_images(model, 20, epoch, z_dim=2, im_size=28, save=True, date=date, frame=idx)

            idx += 1

        # average training losses over all batches and append to lists for plotting later 
        train_loss /= len(train_loader.dataset)
        training_losses.append(train_loss)

        train_recon_loss /= len(train_loader.dataset)
        training_recon_losses.append(train_recon_loss)

        train_kl_loss /= len(train_loader.dataset)
        training_kl_losses.append(train_kl_loss)

        # Validation phase
        # set model to evaluation mode 
        model.eval()
        # initialize loss trackers 
        val_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        
        # loop through validation data loader 
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc="Validation"):
                # move images to the appropriate device 
                images = images.to(device)
                # reconstruct images 
                recon_images, encoding = model(images)

                # compute reconstruction loss. KL-divergence loss and sum to ELBO loss   
                recon_loss = reconstruction_loss(recon_images, images)
                kl_loss = kl_divergence_loss(encoding)
                loss = recon_loss + kl_loss

                # accumulate losses 
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()


        # average training losses over all batches and append to lists for plotting later
        val_loss /= len(test_loader.dataset)
        val_recon_loss /= len(test_loader.dataset)
        val_kl_loss /= len(test_loader.dataset)

        validation_losses.append(val_loss)
        validation_recon_losses.append(val_recon_loss)
        validation_kl_losses.append(val_kl_loss)

        # print training and validation loss for current epoch 
        print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Visualization and Reconstruction phase 
        # load images for visualization, use same images for every epoch
        images, _ = next(iter(test_loader))[:8]
        # reconstruct images 
        recon_images, _ = model(images)

        # visualize images and reconstructed images
        show_images_grid(images.cpu(), date, title=f'Input images after epoch {epoch + 1}', save=True)
        show_images_grid(recon_images.cpu(), date, title=f'Reconstructed images after epoch {epoch + 1}', save=True)

        # in the second and all further epochs and for latent dimension 2 plot the latent space  
        if z_dim == 2 and epoch != 0:
            plot_latent_images(model, 20, epoch, z_dim=2, im_size=28, save=True, date=date)

    return model, training_losses, validation_losses, training_recon_losses, validation_recon_losses, training_kl_losses, validation_kl_losses



# TRAINING  
# ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    # set dimensionality of the latent space 
    z_dim = 8
    # determine the device to use (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # number of training epochs 
    train_epochs = 20
    
    # date and time for model name 
    date = time.strftime("%Y%m%d-%H%M%S")
    model_name = 'vae_' + str(z_dim) + '_' + date

    # create a folder to store results, named using the current date and time
    os.makedirs(date, exist_ok=True)

    # create in the folder a new subfolder images
    os.makedirs(date + '/images', exist_ok=True)

    # if model exists, load it else train it
    try:
        vae = VAE(z_dim=z_dim)
        vae.load_state_dict(torch.load(model_name + '.pth'))
    except FileNotFoundError:
        # if the model is not found, train a new model
        vae, train_losses, val_losses, train_recon_losses, val_recon_losses, train_kl_losses, val_kl_losses = train_model(date, epochs=train_epochs, z_dim=z_dim)
        # save the trained model's weights to a file 
        torch.save(vae.state_dict(), date + '/' + model_name + '.pth')

        # save training and validation losses as .npy files 
        np.save(date + '/' + 'train_losses_' + model_name + '.npy', np.array(train_losses))
        np.save(date + '/' + 'val_losses_' + model_name + '.npy', np.array(val_losses))

        np.save(date + '/' + 'train_recon_losses_' + model_name + '.npy', np.array(train_recon_losses))
        np.save(date + '/' + 'val_recon_losses_' + model_name + '.npy', np.array(val_recon_losses))

        np.save(date + '/' + 'train_kl_losses_' + model_name + '.npy', np.array(train_kl_losses))
        np.save(date + '/' + 'val_kl_losses_' + model_name + '.npy', np.array(val_kl_losses))


    # plot reconstruction loss, KL-divergence loss and total loss individually 
    plot_losses(val_losses, train_losses, date + "/losses", title='ELBO Losses')
    plot_losses(val_recon_losses, train_recon_losses, date + "/recon_losses", title='Reconstruction Losses')
    plot_losses(val_kl_losses, train_kl_losses, date + "/kl_losses", title='KL Divergence Losses')

    # show some original and reconstructed test images
    for images, _ in test_loader:
        images = images.to(device)
        recon_images, encoding = vae(images)
        show_images_grid(images.cpu(), date, title=f'Input images')
        show_images_grid(recon_images.cpu(), date, title=f'Reconstructed images')
        break