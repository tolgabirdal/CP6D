import torch
import torch.nn.functional as F

def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    est = torch.cat((est_pose[:, 0:3], est_pose_q), 1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))

    if torch.abs(inner_prod) <= 1:
        orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / torch.pi
    else:
        origin = torch.abs(torch.abs(inner_prod) - int(torch.abs(inner_prod)) - 1)
        orient_err = 2 * torch.acos(origin) * 180 / torch.pi
    return posit_err, orient_err

