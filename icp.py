import numpy as np
import torch
from IPython import embed
import torch.nn.functional as F

def rot_err(est_pose, gt_pose):

    est_pose_q = F.normalize(est_pose, p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose, p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1)) 
    # if torch.abs(inner_prod) <= 1:
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / torch.pi
    # else:
    #     origin = torch.abs(torch.abs(inner_prod) - int(torch.abs(inner_prod)) - 1)
    #     orient_err = 2 * torch.acos(origin) * 180 / torch.pi
    return orient_err


class ICP_ROT:
    def __init__(self, gt_rot, pred_rot):
        self.gt = gt_rot
        self.pred = pred_rot
        self.non_conformity_scores = []
        self.alpha = 0.05
    
    def compute_non_conformity_scores(self):
        self.non_conformity_scores = rot_err(self.gt, self.pred).squeeze()
        return self.non_conformity_scores
    
    def get_non_conformity_scores(self):
        return torch.tensor(self.non_conformity_scores)
    
    def compute_p_value(self, nc_score):
        p_value = 0
        for i in range(self.gt.shape[0]):
            if self.non_conformity_scores[i] >= nc_score:
                p_value += 1
        return p_value/self.gt.shape[0]
    
    def sample_pose(self, sample_iter=1000):
        # Sample pose from the ground truth poses
        
        # Randomly generate quaternion components

        quats = torch.randn(sample_iter, 4)

        # Normalize each quaternion to unit length
        norm_quats = quats / torch.norm(quats, dim=1).unsqueeze(1)
        return norm_quats
    
    def compute_p_value_from_sampled_poses(self, new_pose):
    # Assuming self.non_conformity_scores is already computed and available
        sampled_poses = self.sample_pose(1000)  # Sampled poses in batch
        sample_nc_scores = rot_err(new_pose.unsqueeze(0), sampled_poses).squeeze()  # Batch compute non-conformity scores

        # Expand dimensions for broadcasting
        calibration_scores = self.non_conformity_scores.unsqueeze(0)
        sample_nc_scores = sample_nc_scores.unsqueeze(1)

        # Count how many calibration scores are >= each sampled score, vectorized
        counts = (calibration_scores >= sample_nc_scores).sum(dim=1)

        # Calculate p-values for each sampled pose
        p_values = counts.float() / (self.gt.shape[0] + 1)

        return p_values
    
    def compute_p_value_from_calibration_poses(self, new_pose):
        # Ensure non_conformity_scores are pre-computed for the calibration set
        self.compute_non_conformity_scores()
 
        # Compute the non-conformity score for the new_pose compared to all calibration poses
        new_pose_nc_scores = rot_err(new_pose.repeat(self.gt.shape[0],1), self.gt).squeeze()
        
        # Vectorize the comparison of new_pose's non-conformity scores against the calibration set
        counts = (self.non_conformity_scores >= new_pose_nc_scores).sum()
        
        # Calculate p-values for the new_pose based on the calibration non-conformity scores
        p_values = counts.float() / (self.gt.shape[0] + 1)
        
        return p_values

    
    