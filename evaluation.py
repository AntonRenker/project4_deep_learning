# load packages 
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import umap.umap_ as umap

# load VAE from model 
from model import VAE


# function to perform dimensionality reduction using UMAP 
def umap_reduction(latents, n_neighbors=15, min_dist=0.1, n_components=2):
    # initialize UMAP model
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    # fit the UMAP model to the latent variables and transform them to reduced dimensions  
    embedding = reducer.fit_transform(latents)
    
    return embedding

# function to extract latent variables and their corresponding labels 
def get_latent(model, data_loader, device):
    # set model to evaluation mode 
    model.eval()
    
    # initialize arrays to store latent variables and labels 
    latents = []
    labels = []

    # disable gradient computation for inference 
    with torch.no_grad():
        for _, (data, label) in enumerate(data_loader):
          # pass data through the encoder to get latent variables 
          mu, _ = model.encoder(data.to(device))
          latents.append(mu.cpu())
          labels.append(label.cpu())
    
    # combine all latents and labels into numpy array 
    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return latents, labels

# function to plot reduced data in 2D with labels 
def plot_reduced_data_2D(reduced_data, test_labels, path=None):
    """
    Plots reduced data points in 2D, colored by their labels.

    Parameters:
    - reduced_data (np.ndarray): Reduced data with 2 dimensions.
    - test_labels (np.ndarray): Labels corresponding to reduced data points.
    - path: path to store image 
    """
    # get unique labels for plotting 
    unique_labels = np.unique(test_labels)
    # generate colormap for the labels 
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Use a colormap with enough colors

    # create the plot 
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        # select data points belonging to the current label 
        idx = test_labels == label
        plt.scatter(
            reduced_data[idx, 0], reduced_data[idx, 1],
            color=colors(i), label=f"Label {label}", s=5, alpha=0.8
        )

    # add plot title, labels, and legend 
    plt.title("2D Plot of Reduced Data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    
    # save the plot and display it 
    if path is not None:
        plt.savefig(path)
    plt.show()


# function to plot reduced data in 3D with labels 
def plot_reduced_data_3D(reduced_data, test_labels, path=None):
    """
    Plots reduced data points in 3D, colored by their labels.

    Parameters:
    - reduced_data (np.ndarray): Reduced data with 3 dimensions.
    - test_labels (np.ndarray): Labels corresponding to data points.
    - path: path to store image 
    """
    # get unique labels for plotting 
    unique_labels = np.unique(test_labels)
    # generate colormap for labels 
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Use a colormap with enough colors

    # create a 3D plot 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(unique_labels):
        # select data points belonging to the current label 
        idx = test_labels == label
        ax.scatter(
            reduced_data[idx, 0], reduced_data[idx, 1], reduced_data[idx, 2],
            color=colors(i), label=f"Label {label}", s=5, alpha=0.8
        )

    # add plot title, axis labels, and legend 
    ax.set_title("3D Plot of Reduced Data")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend()

    # save or display the plot 
    if path is not None:
        plt.savefig(path)
    plt.show()

# function to calculate the center of each class in latent space 
def get_class_center(latents, labels):
    """
    Get the center of each class in the latent space.

    Parameters:
    - latents (np.ndarray): Latent variables of the data points.
    - labels (np.ndarray): Labels corresponding to data points.

    Returns:
    - class_centers (np.ndarray): Centers of each class in the latent space.
    """
    # get unique class labels 
    unique_labels = np.unique(labels)
    # initialize list for class centers 
    class_centers = []

    for label in unique_labels:
        # calculate the mean latent vector for the current label 
        idx = labels == label
        class_centers.append(np.mean(latents[idx], axis=0))
    
    return np.array(class_centers)

# function to get interpolated points between two class centers 
def interpolate_2_class_centers(class_centers, n_steps=10):
    """
    Interpolates between two class centers in the latent space.

    Parameters:
    - class_centers (np.ndarray): Centers of the two classes.
    - n_steps (int): Number of interpolation steps.

    Returns:
    - interpolations (np.ndarray): Interpolated points between the two class centers.
    """
    start, end = class_centers

    # initialize array for interpolation 
    interpolations = np.zeros((n_steps, start.shape[0]))
    # generate interpolated points 
    for i in range(n_steps):
        interpolations[i] = (start * (n_steps - i - 1) + end * (i + 1)) / n_steps

    return interpolations

# function to generate images from interpolated latent points 
def generate_picture_from_interpolation(model, interpolations, path=None):
    """
    Generates pictures from the interpolated points in the latent space.

    Parameters:
    - model (nn.Module): VAE model.
    - interpolations (np.ndarray): Interpolated points in the latent space.
    - path: path to save image 
    """
    # set model to evaluation mode 
    model.eval()
    # disable gradient computation 
    with torch.no_grad():
        # convert interpolations to a tensor and pass through the decoder 
        interpolations = torch.tensor(interpolations).float()
        interpolations = model.decoder(interpolations).cpu().numpy()

    # olot the generated images 
    plt.figure(figsize=(10, 8))
    for i, img in enumerate(interpolations):
        plt.subplot(1, len(interpolations), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

    # save or display images 
    if path is not None:
        plt.savefig(path)
    plt.show()
    

# DIMENSION REDUCTION 
# ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    # latent space dimension 
    z_dim = 8
    # path for model and to save images of dimension reduction 
    model_path = "20250112-155320/vae_8_20250112-155320.pth"
    save_path = "20250112-155320/evaluation/"
    # load the model
    try:
        vae = VAE(z_dim=z_dim)
        vae.load_state_dict(torch.load(model_path))
    except:
        print("Model not found")

    # load and transform MNIST data 
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST('.', download=True, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    # load and otherwise generate variables and labels 
    try:
        latent_variables = np.load(save_path + '/data/latents.npy')
        lables = np.load(save_path + '/data/labels.npy')
        latents_2d = np.load(save_path + '/data/latents_2d.npy')
        latents_3d = np.load(save_path + '/data/latents_3d.npy')
    except:
        latent_variables, lables = get_latent(vae, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # safe latent variables
        np.save(save_path + '/data/latents.npy', latent_variables)

        # safe labels
        np.save(save_path + '/data/labels.npy', lables)
        
        # reduce dimensions to 2D using UMAP
        latents_2d = umap_reduction(latent_variables, n_neighbors=15, min_dist=0.1, n_components=2)

        # reduce dimensions to 3D using UMAP
        latents_3d = umap_reduction(latent_variables, n_neighbors=15, min_dist=0.1, n_components=3)

        # safe reduced latent variables
        np.save(save_path + '/data/latents_2d.npy', latents_2d)
        np.save(save_path + '/data/latents_3d.npy', latents_3d)

    # plot reduced data in 2D
    # plot_reduced_data_2D(latents_2d, lables, path=save_path + 'umap_2D.png')

    # plot reduced data in 3D
    # plot_reduced_data_3D(latents_3d, lables, path=save_path + 'umap_3D.png')

    # compute class centers in the latent space
    class_centers = get_class_center(latent_variables, lables)

    # interpolate all combinations of class centers 
    for i in range(len(class_centers)):
        for j in range(i + 1, len(class_centers)):
            # select two class centers 
            two_class_centers = (class_centers[i], class_centers[j])

            # interpolate between these class centers
            interpolations = interpolate_2_class_centers(two_class_centers)

            # generate pictures from the interpolated points
            path = save_path + f'interpolations_{i}_{j}.png'
            generate_picture_from_interpolation(vae, interpolations, path=path)