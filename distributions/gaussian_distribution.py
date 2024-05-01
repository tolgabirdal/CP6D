import numpy as np
import torch

class GaussianUncertainty():
    def __init__(self, pred_region, new_pose) -> None:
        super(GaussianUncertainty, self).__init__()
        
        # Convert input to tensors
        pred_region = torch.tensor(pred_region, dtype=torch.float32)
        
        # Normalization of the prediction region
        self.mean = torch.mean(pred_region, dim=0)
        self.std = torch.std(pred_region, dim=0)
        self.pred_region_normalized = (pred_region - self.mean) / self.std
        
        # Normalize new pose using the same mean and std
        self.new_pose = torch.tensor(new_pose, dtype=torch.float32)
        self.new_pose_normalized = (self.new_pose - self.mean) / self.std
        
        # Ensure the new_pose_normalized is flattened (if necessary)
        if self.new_pose_normalized.ndim > 1:
            self.new_pose_normalized = self.new_pose_normalized.flatten()
        
        # Calculate the covariance matrix of the normalized prediction region
        self.cov = torch.cov(self.pred_region_normalized.T)  # Ensure correct dimensionality for covariance

    def compute_entropy(self):
        # Regularization term: small value added to the diagonal to ensure the matrix is invertible
        epsilon = 1e-5
        regularized_cov = self.cov + torch.eye(self.cov.shape[0]) * epsilon
        
        # Calculate determinant of the covariance matrix
        det_cov = torch.det(regularized_cov)
        
        # Number of dimensions (should be 3 for tx, ty, tz)
        n = regularized_cov.shape[0]
        
        # Compute entropy
        entropy = 0.5 * torch.log((2 * np.pi * np.e) ** n * det_cov)
        
        return entropy.item()

# Example usage:
pred_region = [[1, 2, 3], [2, 3, 3], [3, 4, 5]]  # Example dataset
new_pose = [[2, 3, 3]]  # New pose to evaluate
gu = GaussianUncertainty(pred_region, new_pose)
entropy = gu.compute_entropy()
print("Entropy (Measure of Uncertainty):", entropy)
