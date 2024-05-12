import torch
from IPython import embed
class GaussianUncertainty():
    def __init__(self, dataset) -> None:
        super(GaussianUncertainty, self).__init__()
        
        self.dataset = dataset

    def compute_uncertainty_score(self, pred_region, new_pose, decay_rate=0.01):
        # if len(pred_region) == 0 or len(pred_region) == 1:
        #     return 1.

        # Calculate the covariance matrix of the normalized prediction region
        cov = torch.cov(pred_region.T)  # Ensure correct dimensionality for covariance
        
        # Regularization term: small value added to the diagonal to prevent singularity
        epsilon = 1e-5
        regularized_cov = cov + torch.eye(cov.shape[0]) * epsilon
        
        # Inverting the covariance matrix with regularization
        cov_inv = torch.linalg.inv(regularized_cov)
        
        # Flatten the new_pose_normalized if necessary
        if new_pose.ndim > 1:
            new_pose = new_pose.flatten()

        # Computing Mahalanobis distance
        diff = new_pose
        mahalanobis_distance = torch.sqrt(torch.dot(diff, torch.mv(cov_inv, diff)))
        # Calculate the uncertainty score using exponential decay
        score = 2. / (1. + torch.exp(-decay_rate * mahalanobis_distance)) - 1.
        return score.item()
    def compute_uncertainty_score_entropy(self, pred_region):
        # if len(pred_region) == 0 or len(pred_region) == 1:
        #     return 1.
        pred_region = pred_region[:, :3]
        if self.dataset == "7Scenes":
            pred_region = pred_region * 100
        cov = torch.cov(pred_region.T)
        entropy = 3 / 2 + 3 / 2 * torch.log(2 * torch.tensor(torch.pi)) + 1 / 2 * torch.log(torch.det(cov))
        uncertainty = 1 / (1 + torch.exp(-entropy)) # sigmoid
        return uncertainty
# Example usage:
if __name__ == '__main__':
    mean = torch.tensor([2, 3, 4])  # Precomputed mean of the data
    std = torch.tensor([1, 1, 1])   # Precomputed standard deviation of the data
    gu = GaussianUncertainty(mean, std)

    pred_region = torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # Example dataset
    new_pose = torch.tensor([[-5, -1, 3]])  # New pose to evaluate as a tensor
    uncertainty_score = gu.compute_uncertainty_score(pred_region, new_pose)
    print("Uncertainty Score:", uncertainty_score)
