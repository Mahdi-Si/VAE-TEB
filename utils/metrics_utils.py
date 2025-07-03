import numpy as np
from numpy.linalg import slogdet
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt


def compute_mutual_information(X, Y, reduce_dim=False, n_components_X=50, n_components_Y=25):
    """
    Compute the mutual information between two sets of signals X and Y.

    Parameters:
    - X: numpy array of shape (N, T, Cx)
        Original signals with N samples, each of length T and Cx channels.
    - Y: numpy array of shape (N, T, Cy)
        Latent representations corresponding to X, with Cy channels.
    - reduce_dim: bool (default: False)
        Whether to apply PCA to reduce dimensionality before computing mutual information.
    - n_components_X: int (default: 50)
        Number of principal components to retain for X if reduce_dim is True.
    - n_components_Y: int (default: 25)
        Number of principal components to retain for Y if reduce_dim is True.

    Returns:
    - mutual_information: float
        Estimated mutual information between X and Y.
    """

    # Ensure input arrays are numpy arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Get the dimensions
    N, T, Cx = X.shape
    _, _, Cy = Y.shape

    # Flatten each sample into a vector
    # X_flat will have shape (N, T * Cx)
    X_flat = X.reshape(N, T * Cx)
    Y_flat = Y.reshape(N, T * Cy)

    if reduce_dim:
        # Apply PCA to X
        pca_X = PCA(n_components=n_components_X, svd_solver='full')
        X_reduced = pca_X.fit_transform(X_flat)

        # Apply PCA to Y
        pca_Y = PCA(n_components=n_components_Y, svd_solver='full')
        Y_reduced = pca_Y.fit_transform(Y_flat)
    else:
        X_reduced = X_flat
        Y_reduced = Y_flat

    # Concatenate X and Y along the feature axis
    XY = np.hstack((X_reduced, Y_reduced))

    # Compute covariance matrices
    # Transpose to shape (features, samples) for np.cov
    cov_X = np.cov(X_reduced, rowvar=False)
    cov_Y = np.cov(Y_reduced, rowvar=False)
    cov_XY = np.cov(XY, rowvar=False)

    # Regularize covariance matrices to ensure positive definiteness
    epsilon = 1e-10
    cov_X += epsilon * np.eye(cov_X.shape[0])
    cov_Y += epsilon * np.eye(cov_Y.shape[0])
    cov_XY += epsilon * np.eye(cov_XY.shape[0])

    # Compute log-determinants
    sign_X, logdet_X = slogdet(cov_X)
    sign_Y, logdet_Y = slogdet(cov_Y)
    sign_XY, logdet_XY = slogdet(cov_XY)

    # Ensure that the determinants are positive
    if sign_X <= 0 or sign_Y <= 0 or sign_XY <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    # Compute mutual information
    mutual_information = 0.5 * (logdet_X + logdet_Y - logdet_XY)

    return mutual_information



def discretize_signal(signal, bins=10):
    """
    Discretize a continuous signal into discrete bins.

    Parameters:
        signal (np.array): The input continuous signal to discretize.
        bins (int): Number of bins to use for discretization.

    Returns:
        np.array: Discretized signal.
    """
    discretized_signal = np.digitize(signal, bins=np.linspace(np.min(signal), np.max(signal), bins))
    return discretized_signal


def calculate_mutual_information(X, Z, bins=10):
    """
    Calculate the mutual information between each channel of the input signal X and latent representation Z.

    Parameters:
        X (np.array): Input signal of shape (samples, length, channels).
        Z (np.array): Latent representation of shape (samples, length, channels).
        bins (int): Number of bins for discretization.

    Returns:
        np.array: Matrix of mutual information values between each channel of X and Z.
    """
    num_X_channels = X.shape[2]
    num_Z_channels = Z.shape[2]
    mi_matrix = np.zeros((num_X_channels, num_Z_channels))

    # Discretize each channel independently
    for i in range(num_X_channels):
        X_discretized = discretize_signal(X[:, :, i].flatten(), bins=bins)
        for j in range(num_Z_channels):
            Z_discretized = discretize_signal(Z[:, :, j].flatten(), bins=bins)
            # Calculate mutual information between X_i and Z_j
            mi_matrix[i, j] = mutual_info_score(X_discretized, Z_discretized)

    return mi_matrix


def visualize_mutual_information(mi_matrix):
    """
    Visualize the mutual information matrix as a heatmap.

    Parameters:
        mi_matrix (np.array): Matrix of mutual information values.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mi_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Mutual Information')
    plt.xlabel('Latent Channels')
    plt.ylabel('Input Channels')
    plt.title('Mutual Information between Input and Latent Channels')
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    num_samples = 1000
    length = 300
    num_X_channels = 11
    num_Z_channels = 5

    # Simulate input signal X and latent representation Z
    X = np.random.rand(num_samples, length, num_X_channels)  # Input signal (random values as example)
    Z = 0.5 * X[:, :, :num_Z_channels] + 0.5 * np.random.rand(num_samples, length,
                                                              num_Z_channels)  # Latent signal with some noise
