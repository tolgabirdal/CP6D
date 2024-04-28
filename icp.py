import numpy as np
import torch
from IPython import embed
import torch.nn.functional as F

# compute the relative pose
def normalize_vector( v):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    return v

def compute_quaternions_from_rotation_matrices(matrices):
    batch=matrices.shape[0]
    
    w=torch.sqrt(torch.max(1.0 + matrices[:,0,0] + matrices[:,1,1] + matrices[:,2,2], torch.zeros(1))) / 2.0
    w = torch.max (w , torch.autograd.Variable(torch.zeros(batch))+1e-8) #batch
    w4 = 4.0 * w
    x= (matrices[:,2,1] - matrices[:,1,2]) / w4
    y= (matrices[:,0,2] - matrices[:,2,0]) / w4
    z= (matrices[:,1,0] - matrices[:,0,1]) / w4
    quats = torch.cat((w.view(batch,1), x.view(batch, 1),y.view(batch, 1), z.view(batch, 1) ), 1   )
    quats = normalize_vector(quats)
    return quats

def compute_rotation_matrix_from_quaternion( quaternion, n_flag=True):
    batch=quaternion.shape[0]
    if n_flag:
        quat = normalize_vector(quaternion)
    else:
        quat = quaternion
    qw = quat[...,0].view(batch, 1)
    qx = quat[...,1].view(batch, 1)
    qy = quat[...,2].view(batch, 1)
    qz = quat[...,3].view(batch, 1)

    # Unit quaternion rotation matrices computatation  
    xx = qx*qx
    yy = qy*qy
    zz = qz*qz
    xy = qx*qy
    xz = qx*qz
    yz = qy*qz
    xw = qx*qw
    yw = qy*qw
    zw = qz*qw

    row0 = torch.cat((1-2*yy-2*zz, 2*xy - 2*zw, 2*xz + 2*yw), 1) #batch*3
    row1 = torch.cat((2*xy+ 2*zw,  1-2*xx-2*zz, 2*yz-2*xw  ), 1) #batch*3
    row2 = torch.cat((2*xz-2*yw,   2*yz+2*xw,   1-2*xx-2*yy), 1) #batch*3
    
    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch,1,3), row2.view(batch,1,3)),1) #batch*3*3
    
    return matrix

def rot_err_q(est_pose, gt_pose):
                 
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

def rot_err_R(est_pose, gt_pose):
    est_R = compute_rotation_matrix_from_quaternion(est_pose)
    gt_R = compute_quaternions_from_rotation_matrices(gt_pose)
    rot = torch.matmul(est_R.transpose(1, 2), gt_R)
    U, S, Vh = torch.linalg.svd(rot)
    V = Vh.mH
    log_rot = U @ torch.diag(torch.log(S)) @ V
    rot_err = torch.mean(torch.abs(log_rot)) / torch.pi
    return rot_err

def translation_err(est_pose, gt_pose):
    """
    Calculate the position error given the estimated and ground truth pose(s).
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    return posit_err


class ICP_ROT:
    def __init__(self, gt_rot, pred_rot):
        super(ICP_ROT, self).__init__()
        self.gt = gt_rot
        self.pred = pred_rot
        self.non_conformity_scores = []
        self.alpha = 0.05
    
    def compute_non_conformity_scores(self):
        self.non_conformity_scores = rot_err_q(self.gt, self.pred).squeeze()
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
        sample_nc_scores = rot_err_q(new_pose.unsqueeze(0), sampled_poses).squeeze()  # Batch compute non-conformity scores

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

    
class Inductive_Conformal_Predcition:
    class ICP_ROT:
        def __init__(self, gt, pred):
            """
            Initialize the ICP_ROT class.

            Parameters:
            - gt (numpy.ndarray): Ground truth data. It should be a 2D array with shape (n, m),
                                  where n is the number of samples and m is the number of features.
            - pred (numpy.ndarray): Predicted data. It should be a 2D array with shape (n, m),
                                    where n is the number of samples and m is the number of features.
            """
            super(ICP_ROT, self).__init__()
            self.gt = gt
            self.pred = pred
            self.gt_rot = gt[:, 3:]
            self.pred_rot = pred[:, 3:]
            self.gt_trans = gt[:, :3]
            self.pred_trans = pred[:, :3]
            self.non_conformity_scores_rot = []
            self.non_conformity_scores_trans = []
            self.alpha = 0.05
            self.rot_err = rot_err_R
            self.trans_err = translation_err
    
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