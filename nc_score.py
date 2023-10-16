import numpy as np
import scipy
from scipy.linalg import logm


def pose_error_quat(est_pose, gt_pose):
    if est_pose.ndim == 1:
        est_pose = est_pose.reshape(1, -1)
    if gt_pose.ndim == 1:
        gt_pose = gt_pose.reshape(1, -1)
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s)
    :param est_pose: (np.ndarray) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (np.ndarray) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = np.linalg.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], axis=1)
    est_pose_q = est_pose[:, 3:] / np.linalg.norm(est_pose[:, 3:], axis=1, keepdims=True)
    gt_pose_q = gt_pose[:, 3:] / np.linalg.norm(gt_pose[:, 3:], axis=1, keepdims=True)
    est = np.concatenate((est_pose[:, 0:3], est_pose_q), axis=1)
    inner_prod = np.matmul(est_pose_q[:, np.newaxis, :], gt_pose_q[:, :, np.newaxis])

    orient_err = 2 * np.arccos(np.abs(inner_prod)) * 180 / np.pi
    orient_err = np.squeeze(orient_err)

    return posit_err, orient_err

def transformation_matrix(rot_mats, trans_vecs):
    """
    Aggregate a batch of rotation matrices and translation vectors into a batch of transformation matrices
    :param rot_mats: (np.ndarray) a batch of 3x3 rotation matrices of shape (N, 3, 3)
    :param trans_vecs: (np.ndarray) a batch of 3-dimensional translation vectors of shape (N, 3)
    :return: (np.ndarray) a batch of 4x4 transformation matrices of shape (N, 4, 4)
    """
    # Reshape the translation vectors to shape (N, 3, 1)
    trans_vecs = np.reshape(trans_vecs, (-1, 3, 1))

    # Create a batch of identity matrices of shape (N, 4, 4)
    trans_mats = np.tile(np.eye(4), (rot_mats.shape[0], 1, 1))

    # Copy the rotation matrices into the upper left 3x3 blocks
    trans_mats[:, :3, :3] = rot_mats

    # Copy the translation vectors into the rightmost columns
    trans_mats[:, :3, 3] = trans_vecs.squeeze()

    return trans_mats

def pose_error_transformation(gt_pose, pred_pose, lamb=0.5):
    """
    Calculate the position and orientation error given the estimated and ground truth poses
    :param gt_pose: (np.ndarray) a batch of ground truth poses of shape (N, 4, 4)
    :param pred_pose: (np.ndarray) a batch of predicted poses of shape (N, 4, 4)
    :return: (float, float) the mean squared error of the translation and orientation errors
    """
    if gt_pose.ndim == 2:
        gt_pose = gt_pose.reshape(1, 4, 4)
    if pred_pose.ndim == 2:
        pred_pose = pred_pose.reshape(1, 4, 4)
    # Extract the ground truth and predicted rotation matrices and translation vectors
    gt_rot = gt_pose[:, :3, :3]
    pred_rot = pred_pose[:, :3, :3]
    gt_trans = gt_pose[:, :3, 3]
    pred_trans = pred_pose[:, :3, 3]
    orient_error = np.zeros(gt_rot.shape[0])
    

    # Compute the translation error as the mean squared error (MSE) between the predicted and ground truth translation vectors
    trans_err = np.linalg.norm(pred_trans - gt_trans, axis=1)
    mat_product = np.matmul(pred_rot.transpose(0,2,1), gt_rot)
    for i in range(gt_rot.shape[0]):
        log_mat_product = logm(mat_product[i])
        # Compute the orientation error as the logarithmic distance between the predicted and ground truth rotation matrices
        orient_error[i] = 1/np.pi * np.mean(np.linalg.norm(log_mat_product))
    return 0.0, orient_error

def non_conformity_value(pred_pose, gt_pose, score):
    """
    Compute the p-value of a predicted pose given a set of ground truth poses and a non-conformity score function
    :param pred_pose: (np.ndarray) the predicted pose, shape (N, 4, 4)
    :param gt_poses: (np.ndarray) the set of ground truth poses in the training set, shape (N, 4, 4)
    :param score: (callable) a non-conformity score function that takes a predicted pose and a ground truth pose as input
                  and returns a non-negative scalar that is smaller if the predicted pose is more similar to the ground truth pose
    :return: non-conformity score of the predicted pose given the ground truth poses shape (N,)
    """

    # Compute the non-conformity scores between the predicted pose and all ground truth poses
    scores = np.zeros(len(gt_pose))
    for i in range(len(gt_pose)):
        trans, ori = score(pred_pose[i], gt_pose[i])
        scores[i] = trans + ori
    return scores