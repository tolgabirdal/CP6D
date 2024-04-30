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
        
        # Calculate the covariance matrix of the normalized prediction region
        self.cov = torch.cov(self.pred_region_normalized.T)  # Ensure correct dimensionality for covariance

    def compute_gaussian_uncertainty(self):
        # Regularization term: small value added to the diagonal
        epsilon = 1e-5
        regularized_cov = self.cov + torch.eye(self.cov.shape[0]) * epsilon
        
        # Inverting the covariance matrix with regularization
        cov_inv = torch.linalg.inv(regularized_cov)
        
        # Difference between new normalized pose and zero (mean of normalized data)
        diff = self.new_pose_normalized
        
        # Computing Mahalanobis distance
        mahalanobis_distance = torch.sqrt(torch.dot(diff, torch.mv(cov_inv, diff)))
        return mahalanobis_distance

    def compute_uncertainty_score(self, decay_rate):
        distance = self.compute_gaussian_uncertainty()
        score = torch.exp(-decay_rate * distance)
        return 1. - score.item()

# Example usage:
pred_region = [[1, 2, 3], [2, 3, 3], [3, 4, 5]]  # Example dataset
new_pose = [2, 3, 3]  # New pose to evaluate
decay_rate = 0.1  # Example decay rate, adjust based on data characteristics
gu = GaussianUncertainty(pred_region, new_pose)
uncertainty_score = gu.compute_uncertainty_score(decay_rate)
print("Uncertainty Score:", uncertainty_score)
