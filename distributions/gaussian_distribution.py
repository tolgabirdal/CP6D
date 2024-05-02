import torch

class GaussianUncertainty():
    def __init__(self, mean, std) -> None:
        super(GaussianUncertainty, self).__init__()
        
        # Store the mean and standard deviation as tensors
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def normalize_data(self, data):
        """ Normalize data using the predefined mean and std. """
        return (data - self.mean) / self.std

    def compute_uncertainty_score(self, pred_region, new_pose, decay_rate=0.001):
        # Normalize prediction region
        pred_region_normalized = self.normalize_data(pred_region)
        
        # Assuming new_pose is already a tensor when passed to the function
        new_pose_normalized = self.normalize_data(new_pose)

        # Calculate the covariance matrix of the normalized prediction region
        cov = torch.cov(pred_region_normalized.T)  # Ensure correct dimensionality for covariance
        
        # Regularization term: small value added to the diagonal to prevent singularity
        epsilon = 1e-5
        regularized_cov = cov + torch.eye(cov.shape[0]) * epsilon
        
        # Inverting the covariance matrix with regularization
        cov_inv = torch.linalg.inv(regularized_cov)
        
        # Flatten the new_pose_normalized if necessary
        if new_pose_normalized.ndim > 1:
            new_pose_normalized = new_pose_normalized.flatten()

        # Computing Mahalanobis distance
        diff = new_pose_normalized
        mahalanobis_distance = torch.sqrt(torch.dot(diff, torch.mv(cov_inv, diff)))
        
        # Calculate the uncertainty score using exponential decay
        score = 1. - torch.exp(-decay_rate * mahalanobis_distance)
        return score.item()

# Example usage:
mean = torch.tensor([2, 3, 4])  # Precomputed mean of the data
std = torch.tensor([1, 1, 1])   # Precomputed standard deviation of the data
gu = GaussianUncertainty(mean, std)

pred_region = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # Example dataset
new_pose = torch.tensor([[-5, -1, 3]])  # New pose to evaluate as a tensor
uncertainty_score = gu.compute_uncertainty_score(pred_region, new_pose)
print("Uncertainty Score:", uncertainty_score)
