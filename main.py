import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt

import icp
from nc_score import transformation_matrix, pose_error_transformation, non_conformity_value, pose_error_quat

from tqdm import tqdm
import tools
import rpmg
import argparse

from IPython import embed

def read_poses(file):
    df = pd.read_csv(file)
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    
    est_poses = np.zeros((n, 7))
    est_poses[:, 0] = df['est_t1'].values
    est_poses[:, 1] = df['est_t2'].values
    est_poses[:, 2] = df['est_t3'].values
    est_poses[:, 3] = df['est_q1'].values
    est_poses[:, 4] = df['est_q2'].values
    est_poses[:, 5] = df['est_q3'].values
    est_poses[:, 6] = df['est_q4'].values
    return poses, est_poses



def standardize_translation_vectors(translation_vectors):
    # Calculate the mean and standard deviation of the translation vectors
    mean = np.mean(translation_vectors, axis=0)
    std = np.std(translation_vectors, axis=0)

    # Normalize the translation vectors
    normalized_vectors = (translation_vectors - mean) / std

    return normalized_vectors, mean, std

# Example usage:
# num_samples = 100
# translation_vectors = np.random.rand(num_samples, 3)  # Replace with actual translation vectors

# normalized_vectors, mean, std = standardize_translation_vectors(translation_vectors)
# print("Normalized translation vectors:", normalized_vectors)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sn", help="name of scenes e.g. chess, fire")
    args = arg_parser.parse_args()
    
    test_gt, test_est = read_poses('./dataset/7Scenes/train_cal_0.5/abs_7scenes_pose.csv_'+args.sn+'_test.csv_mstransformer_pred.csv')
    calib_gt, calib_est = read_poses('./dataset/7Scenes/train_cal_0.5/7Scenes_subset_'+args.sn+'_calib.csv_mstransformer_pred.csv')
    
    # Split Validation Set
    # np.random.seed(686)
    # rate = 0.5
    # idx = np.random.permutation(len(chess_gt))
    # calibrate = idx[:int(idx.size * rate)]
    # test = idx[int(idx.size * rate):]

    # Get Standardised Transition
    # TODO: [-1,1] Standard
    cal_gt_trans, cal_gt_mean, cal_gt_std = standardize_translation_vectors(calib_gt[:, 0:3])
    calib_gt_trans = (calib_gt[:, 0:3] - cal_gt_mean) / cal_gt_std
    calib_pred_trans = (calib_est[:, 0:3] - cal_gt_mean) / cal_gt_std
    
    test_gt_trans = (test_gt[:, 0:3] - cal_gt_mean) / cal_gt_std
    test_pred_trans = (test_est[:, 0:3] - cal_gt_mean) / cal_gt_std

    # Get Rotation
    calib_gt_rot = tools.compute_rotation_matrix_from_quaternion(torch.tensor(calib_gt[:, 3:]), n_flag=False).numpy()
    calib_pred_q = calib_est[:, 3:] / np.linalg.norm(calib_est[:, 3:], axis=1, keepdims=True)
    calib_pred_rot = tools.compute_rotation_matrix_from_quaternion(torch.tensor(calib_est), n_flag=False).numpy()
    
    test_gt_rot = tools.compute_rotation_matrix_from_quaternion(torch.tensor(test_gt[:, 3:]), n_flag=False).numpy()
    test_pred_q = test_est[:, 3:] / np.linalg.norm(test_est[:, 3:], axis=1, keepdims=True)
    test_pred_rot = tools.compute_rotation_matrix_from_quaternion(torch.tensor(test_est), n_flag=False).numpy()
    
    # Get Transformation Matrix
    calib_gt_T = transformation_matrix(calib_gt_rot, calib_gt_trans)
    calib_pred_T = transformation_matrix(calib_pred_rot, calib_pred_trans)
    
    test_gt_T = transformation_matrix(test_gt_rot, test_gt_trans)
    test_pred_T = transformation_matrix(test_pred_rot, test_pred_trans)
    
    non_conformity_score = non_conformity_value(calib_pred_T, calib_gt_T, pose_error_transformation)
    # p_values = p_value(chess_pred_T[test[0]], chess_gt_T[calibrate], pose_error_transformation, non_conformity_score)
    print(non_conformity_score.shape)
    p_values_all, pred_all_region = [], []
    for alpha in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        for i in tqdm(range(len(test_pred_T))):
            p_values, predicted_region = icp.inductive_conformal_prediction(test_pred_T[i], calib_gt_T, non_conformity_score, alpha=alpha)
            tqdm.write(str(predicted_region.shape))
            p_values_all.append(p_values); pred_all_region.append(predicted_region)
        np.save('Pred_Region/7Scenes/'+args.sn+'/pred_all_region_alpha_'+str(alpha),pred_all_region)