import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
from scipy.linalg import pinvh
from sklearn.manifold import MDS

from model import VAE


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

def construct_mutual_knn_graph(X, k=5):
    """
    Construct a mutual k-NN graph and weight edges using a Gaussian kernel.

    Parameters:
        X (np.ndarray): Data matrix where rows are samples.
        k (int): Number of neighbors for k-NN.
        sigma (float): Gaussian kernel parameter.

    Returns:
        adj_matrix (np.ndarray): Symmetric weighted adjacency matrix.
    """

    # Compute k-NN graph (sparse adjacency matrix)
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    knn_distances = knn_graph.toarray()

    # Symmetrize to get mutual k-NN graph
    mutual_knn = np.maximum(knn_distances, knn_distances.T)

    # return adj_matrix
    return mutual_knn

def is_graph_connected(adj_matrix):
    # Use scipy's connected_components to check connectivity
    n_components, _ = connected_components(csgraph=adj_matrix, directed=False)
    return n_components == 1

def compute_commute_time_distance(knn_graph):
    """
    Compute the Commute Time Distance for a given k-NN graph.

    Parameters:
        knn_graph (np.ndarray): Adjacency matrix of the k-NN graph (NxN).
                                The graph should be symmetric and connected.

    Returns:
        np.ndarray: Commute time distance matrix (NxN).
    """
    # Ensure the graph is symmetric
    if not np.allclose(knn_graph, knn_graph.T):
        raise ValueError("Input graph must be symmetric.")
    
    # Compute the degree matrix
    degrees = np.sum(knn_graph, axis=1)
    degree_matrix = np.diag(degrees)

    # Compute the Laplacian matrix
    laplacian = degree_matrix - knn_graph

    # Compute the pseudoinverse of the Laplacian
    laplacian_pinv = pinvh(laplacian)  # Use pinvh for symmetric matrices

    # Vectorized computation of commute time distance
    diag_elements = np.diag(laplacian_pinv)
    ctd_matrix = diag_elements[:, None] + diag_elements[None, :] - 2 * laplacian_pinv

    # Return the CTD matrix
    return ctd_matrix

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

if __name__ == "__main__":
    z_dim = 8
    folder = '20250112-193429'
    
    # create folder evaluation
    if not os.path.exists(folder + '/evaluation'):
        os.makedirs(folder + '/evaluation')
        os.makedirs(folder + '/evaluation/data')


    model_path = folder + "/vae_8_20250112-193429.pth"
    save_path = folder + "/evaluation/"

    # load model
    try:
        vae = VAE(z_dim=z_dim)
        vae.load_state_dict(torch.load(model_path))
    except:
        print("Model not found")
        # end program
        exit()

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST('.', download=True, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    # get latent variables
    try:
        latent_variables = np.load(save_path + '/data/latents.npy')
        lables = np.load(save_path + '/data/labels.npy')
    except:
        latent_variables, lables = get_latent(vae, test_loader,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
        
    # only use 1000  random samples
    idx = np.random.choice(len(latent_variables), 1000, replace=False)
    latent_variables = latent_variables[idx]
    lables = lables[idx]

    # construct mutual k-NN graph
    k=3
    knn_latent = construct_mutual_knn_graph(latent_variables, k=k)

    # increase k until the graph is connected
    while not is_graph_connected(knn_latent):
        print('Graph is not connected, increasing k')
        k +=1
        knn_latent = construct_mutual_knn_graph(latent_variables, k=k)
    
    print(f'k={k} is the smallest k for which the graph is connected')

    # compute commute time distance
    try:
        ctd_latent = np.load(save_path + '/data/ctd_latent.npy')
        print("Commute Time Distance matrix loaded")
    # compute commute time distance
    except:
        ctd_latent = compute_commute_time_distance(knn_latent)
        # save ctd_latent
        np.save(save_path + '/data/ctd_latent.npy', ctd_latent)
    
    print("Commute Time Distance matrix computed")

    # Reduce dimensions to 2D and 3D using MDS
    try:
        reduced_latent_ctd_2D = np.load(save_path + '/data/reduced_latent_ctd_2D.npy')
        reduced_latent_ctd_3D = np.load(save_path + '/data/reduced_latent_ctd_3D.npy')
        print("Reduced data loaded")
    except:
        mds_2D = MDS(n_components=2, dissimilarity='precomputed')
        reduced_latent_ctd_2D = mds_2D.fit_transform(ctd_latent)
        np.save(save_path + '/data/reduced_latent_ctd_2D.npy', reduced_latent_ctd_2D)

        mds_3D = MDS(n_components=3, dissimilarity='precomputed')
        reduced_latent_ctd_3D = mds_3D.fit_transform(ctd_latent)
        np.save(save_path + '/data/reduced_latent_ctd_3D.npy', reduced_latent_ctd_3D)

    # Plot reduced data in 2D
    plot_reduced_data_2D(reduced_latent_ctd_2D, lables, path=save_path + 'ctd_2D.png')

    # Plot reduced data in 3D
    plot_reduced_data_3D(reduced_latent_ctd_3D, lables, path=save_path + 'ctd_3D.png')
    
