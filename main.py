import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from icp import ICP_ROT
from nc_score import pose_err
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
    return torch.tensor(poses), torch.tensor(est_poses)



def standardize_translation_vectors(translation_vectors):
    # Calculate the mean and standard deviation of the translation vectors
    std, mean = torch.std_mean(translation_vectors, dim=0)

    # Normalize the translation vectors
    normalized_vectors = (translation_vectors - mean) / std

    return normalized_vectors, mean, std


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sn", help="name of scenes e.g. chess, fire")
    args = arg_parser.parse_args()
    
    test_gt, test_est = read_poses('./dataset/7Scenes/train_cal_0.5/abs_7scenes_pose.csv_'+args.sn+'_test.csv_mstransformer_pred.csv')
    calib_gt, calib_est = read_poses('./dataset/7Scenes/train_cal_0.5/7Scenes_subset_'+args.sn+'_calib.csv_mstransformer_pred.csv')
    
    cal_gt_trans, cal_gt_mean, cal_gt_std = standardize_translation_vectors(calib_gt[:, 0:3])
    calib_gt_trans = (calib_gt[:, 0:3] - cal_gt_mean) / cal_gt_std
    calib_pred_trans = (calib_est[:, 0:3] - cal_gt_mean) / cal_gt_std
    
    test_gt_trans = (test_gt[:, 0:3] - cal_gt_mean) / cal_gt_std
    test_pred_trans = (test_est[:, 0:3] - cal_gt_mean) / cal_gt_std

    # Get Rotation
    calib_gt_rot = calib_gt[:, 3:]
    calib_pred_rot = calib_est[:, 3:]
    
    test_gt_rot = test_gt[:, 3:]
    test_pred_rot = test_est[:, 3:]
    
    print("calib gt rot:", calib_gt_rot.shape)
    # print("calib pred rot:", calib_pred_rot.shape)
    # print("test gt rot:", test_gt_rot.shape)
    # print("test pred rot:", test_pred_rot.shape)
    print("calib_gt_trans:", cal_gt_trans.shape)
    
    icp_rot = ICP_ROT(calib_gt_rot, calib_pred_rot)
    for i in range(len(test_pred_rot)):
        new_pose = test_pred_rot[i]
        
        p_value = icp_rot.compute_p_value_from_calibration_poses(new_pose)
        
        print(i, p_value)
    print("finished")