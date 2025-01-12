# load packages 
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import umap.umap_ as umap

# load VAE from model 
from model import VAE


def umap_reduction(latents, n_neighbors=15, min_dist=0.1, n_components=2):
    # Initialize UMAP model
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embedding = reducer.fit_transform(latents)
    return embedding

def get_latent(model, data_loader, device):
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for _, (data, label) in enumerate(data_loader):
          mu, _ = model.encoder(data.to(device))
          latents.append(mu.cpu())
          labels.append(label.cpu())

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    return latents, labels

def plot_reduced_data_2D(reduced_data, test_labels, path=None):
    """
    Plots reduced data points in 2D, colored by their labels.

    Parameters:
        reduced_data (np.ndarray): Reduced data with 2 dimensions.
        test_labels (np.ndarray): Labels corresponding to data points.
    """
    unique_labels = np.unique(test_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Use a colormap with enough colors

    plt.figure(figsize=(10, 8))

    for i, label in enumerate(unique_labels):
        idx = test_labels == label
        plt.scatter(
            reduced_data[idx, 0], reduced_data[idx, 1],
            color=colors(i), label=f"Label {label}", s=5, alpha=0.8
        )

    plt.title("2D Plot of Reduced Data")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    
    if path is not None:
        plt.savefig(path)

    plt.show()

def plot_reduced_data_3D(reduced_data, test_labels, path=None):
    """
    Plots reduced data points in 3D, colored by their labels.

    Parameters:
        reduced_data (np.ndarray): Reduced data with 3 dimensions.
        test_labels (np.ndarray): Labels corresponding to data points.
    """
    unique_labels = np.unique(test_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Use a colormap with enough colors

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, label in enumerate(unique_labels):
        idx = test_labels == label
        ax.scatter(
            reduced_data[idx, 0], reduced_data[idx, 1], reduced_data[idx, 2],
            color=colors(i), label=f"Label {label}", s=5, alpha=0.8
        )

    ax.set_title("3D Plot of Reduced Data")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend()

    if path is not None:
        plt.savefig(path)

    plt.show()


def get_class_center(latents, labels):
    """
    Get the center of each class in the latent space.

    Parameters:
        latents (np.ndarray): Latent variables of the data points.
        labels (np.ndarray): Labels corresponding to data points.

    Returns:
        class_centers (np.ndarray): Centers of each class in the latent space.
    """
    unique_labels = np.unique(labels)
    class_centers = []

    for label in unique_labels:
        idx = labels == label
        class_centers.append(np.mean(latents[idx], axis=0))

    return np.array(class_centers)

def interpolate_2_class_centers(class_centers, n_steps=10):
    """
    Interpolates between two class centers in the latent space.

    Parameters:
        class_centers (np.ndarray): Centers of the two classes.
        n_steps (int): Number of interpolation steps.

    Returns:
        interpolations (np.ndarray): Interpolated points between the two class centers.
    """
    start, end = class_centers
    interpolations = np.zeros((n_steps, start.shape[0]))

    for i in range(n_steps):
        interpolations[i] = (start * (n_steps - i - 1) + end * (i + 1)) / n_steps

    return interpolations

def generate_picture_from_interpolation(model, interpolations, path=None):
    """
    Generates pictures from the interpolated points in the latent space.

    Parameters:
        model (nn.Module): VAE model.
        interpolations (np.ndarray): Interpolated points in the latent space.
    """
    model.eval()
    with torch.no_grad():
        interpolations = torch.tensor(interpolations).float()
        interpolations = model.decoder(interpolations).cpu().numpy()

    plt.figure(figsize=(10, 8))

    for i, img in enumerate(interpolations):
        plt.subplot(1, len(interpolations), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')

    if path is not None:
        plt.savefig(path)

    plt.show()
    


if __name__ == "__main__":
    z_dim = 8
    model_path = "20250112-155320/vae_8_20250112-155320.pth"
    save_path = "20250112-155320/evaluation/"
    try:
        vae = VAE(z_dim=z_dim)
        vae.load_state_dict(torch.load(model_path))
    except:
        print("Model not found")

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST('.', download=True, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    try:
        latent_variables = np.load(save_path + '/data/latents.npy')
        lables = np.load(save_path + '/data/labels.npy')
        latents_2d = np.load(save_path + '/data/latents_2d.npy')
        latents_3d = np.load(save_path + '/data/latents_3d.npy')
    except:
        latent_variables, lables = get_latent(vae, test_loader,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # safe latent variables
        np.save(save_path + '/data/latents.npy', latent_variables)

        #safe labels
        np.save(save_path + '/data/labels.npy', lables)
        
        # Reduce dimensions to 2D using UMAP
        latents_2d = umap_reduction(latent_variables, n_neighbors=15, min_dist=0.1, n_components=2)

        # Reduce dimensions to 3D using UMAP
        latents_3d = umap_reduction(latent_variables, n_neighbors=15, min_dist=0.1, n_components=3)

        # safe reduced latent variables
        np.save(save_path + '/data/latents_2d.npy', latents_2d)
        np.save(save_path + '/data/latents_3d.npy', latents_3d)

    # Plot reduced data in 2D
    # plot_reduced_data_2D(latents_2d, lables, path=save_path + 'umap_2D.png')

    # Plot reduced data in 3D
    # plot_reduced_data_3D(latents_3d, lables, path=save_path + 'umap_3D.png')


    # Compute class centers in the latent space
    class_centers = get_class_center(latent_variables, lables)

    # Interpolate all combinations of class centers
    for i in range(len(class_centers)):
        for j in range(i + 1, len(class_centers)):
            two_class_centers = (class_centers[i], class_centers[j])

            # Interpolate between two class centers
            interpolations = interpolate_2_class_centers(two_class_centers)

            # Generate pictures from the interpolated points
            path = save_path + f'interpolations_{i}_{j}.png'
            generate_picture_from_interpolation(vae, interpolations, path=path)