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




    
class Inductive_Conformal_Predcition:
    def __init__(self, gt, pred, mode, rate=None):
        """
        Initialize the ICP_ROT class.

        Parameters:
        - gt (numpy.ndarray): Ground truth data. It should be a 2D array with shape (n, m),
                                where n is the number of samples and m is the number of features.
        - pred (numpy.ndarray): Predicted data. It should be a 2D array with shape (n, m),
                                where n is the number of samples and m is the number of features.
        """
        super(Inductive_Conformal_Predcition, self).__init__()
        self.gt = gt
        self.pred = pred
        self.gt_rot = gt[:, 3:]
        self.pred_rot = pred[:, 3:]
        self.gt_trans = gt[:, :3]
        self.pred_trans = pred[:, :3]
        self.non_conformity_scores_rot = []
        self.non_conformity_scores_trans = []
        self.alpha = 0.05
        self.rot_err = rot_err_q
        self.trans_err = translation_err
        self.mode = mode
        self.rate = rate
        
        self.nc_scores = None
        self.non_conformity_scores = None
    
    def compute_non_conformity_scores(self):
        self.non_conformity_scores_rot = self.rot_err(self.gt_rot, self.pred_rot).squeeze()
        self.non_conformity_scores_trans = self.trans_err(self.gt_trans, self.pred_trans).squeeze()
        self.non_conformity_scores = {
            "Trans": self.non_conformity_scores_trans,
            "Rot": self.non_conformity_scores_rot
        }
        return self.non_conformity_scores
    
    def compute_nc_scores(self):
        if self.non_conformity_scores == None:
            self.compute_non_conformity_scores()
        if self.mode == "Rot":
            nc_scores = self.non_conformity_scores["Rot"]
        elif self.mode == "Trans":
            nc_scores = self.non_conformity_scores["Trans"]
        elif self.mode == "Combine":
            if rate is None:
                rate = 1.0
                raise Warning("Rate is not provided. Default rate is set to 1.0")
            nc_scores = self.non_conformity_scores["Rot"] + rate * self.non_conformity_scores["Trans"]
            
            print("----------------------------------")
            print("Rate: ", rate, "Mean Rot Error: ", self.non_conformity_scores["Rot"].mean(), "Mean Trans Error: ", self.non_conformity_scores["Trans"].mean())
            print("----------------------------------")
        else:
            raise ValueError("Invalid mode. Please choose from 'Rot', 'Trans' and 'Combine'")
        
        self.nc_scores = nc_scores
    
    def compute_p_value_from_calibration_poses(self, new_pose, p=0.5):
        if self.nc_scores == None:
            self.compute_nc_scores()
        new_pose_q = new_pose[:, 3:]
        new_pose_t = new_pose[:, :3]
 
        # Compute the non-conformity score for the new_pose compared to all calibration poses
        new_pose_nc_scores_rot = rot_err_q(new_pose_q.repeat(self.gt.shape[0],1), self.gt_rot).squeeze()
        new_pose_nc_scores_trans = translation_err(new_pose_t.repeat(self.gt.shape[0],1), self.gt_trans).squeeze()
        
        if self.mode == "Rot":
            new_pose_nc_scores = new_pose_nc_scores_rot
        elif self.mode == "Trans":
            new_pose_nc_scores = new_pose_nc_scores_trans
        elif self.mode == "Combine":
            new_pose_nc_scores = new_pose_nc_scores_rot + self.rate * new_pose
        else:
            raise ValueError("Invalid mode. Please choose from 'Rot','Trans' and 'Combine'")
        
        pred_region = []
        for idx, new_nc_score in enumerate(new_pose_nc_scores):
            p_value = (self.nc_scores >= new_nc_score).float().mean()
            print("P-value: ", p_value)
            if p_value >= p:
                pred_region.append(idx)
        # print("P-value Num: ", len(pred_region))
        return pred_region