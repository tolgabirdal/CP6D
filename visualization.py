import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_bingham(M, Z, num_samples=1000):
    """
    Generate samples from a Bingham distribution (simplified version).
    M: Orthogonal matrix of principal axes.
    Z: Diagonal matrix with concentration parameters.
    """
    # Dimension check
    d = M.shape[0]
    if M.shape[1] != d or Z.shape[0] != d or Z.shape[1] != d:
        raise ValueError("Dimensions of M and Z must match and be square")

    # Sample from a multivariate normal distribution
    Y = np.random.multivariate_normal(np.zeros(d), np.eye(d), num_samples)

    # Normalize to lie on the unit sphere
    Y /= np.linalg.norm(Y, axis=1)[:, np.newaxis]

    # Weight the samples
    weights = np.exp(np.diag(Y @ M @ Z @ M.T @ Y.T))
    weighted_samples = Y.T * weights

    return weighted_samples

def plot_bingham(samples):
    """ Plot samples on a 3D sphere. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[0, :], samples[1, :], samples[2, :])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Visualization of Approximated Bingham Distribution')
    plt.show()

# Define the parameters (example)
M = np.eye(3)  # principal axes
Z = np.diag([-10, -10, 0])  # concentration parameters

# Generate and plot samples
samples = sample_bingham(M, Z, num_samples=1000)
plot_bingham(samples)
