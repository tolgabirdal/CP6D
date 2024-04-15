import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
# Example data: n camera poses with [x, y, z, qx, qy, qz, qw]
# Here we'll just create a dummy array for illustration purposes
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
    return poses

# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qxqy, qxqz, qxqw, qyqz, qyqw, qzqw = qx * qy, qx * qz, qx * qw, qy * qz, qy * qw, qz * qw
    
    R = np.array([[1 - 2*(qy2 + qz2), 2*(qxqy - qzqw), 2*(qxqz + qyqw)],
                  [2*(qxqy + qzqw), 1 - 2*(qx2 + qz2), 2*(qyqz - qxqw)],
                  [2*(qxqz - qyqw), 2*(qyqz + qxqw), 1 - 2*(qx2 + qy2)]])
    return R

def normalize_positions(poses):
    xyz = poses[:, :3]
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    normalized_xyz = 2 * (xyz - min_xyz) / (max_xyz - min_xyz) - 1
    return np.hstack([normalized_xyz, poses[:, 3:]])

# sns = ['heads', 'pumpkin', 'redkitchen', 'stairs']
# sns = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']
sns = ['colosseum_exterior', 'notre_dame_front_facade', 'trevi_fountain']
for sn in tqdm(sns):
    # camera_poses_train = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_train.csv_results.csv')
    # camera_poses_calib = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_cal.csv_results.csv')
    # camera_poses_test = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_test.csv_results.csv')
    # camera_poses_train = read_poses('/home/runyi/Project/TBCP6D/dataset/CambridgeLandmarks_0.5/abs_cambridge_pose_sorted.csv_'+sn+'_train.csv_results.csv')
    # camera_poses_calib = read_poses('/home/runyi/Project/TBCP6D/dataset/CambridgeLandmarks_0.5/abs_cambridge_pose_sorted.csv_'+sn+'_cal.csv_results.csv')
    # camera_poses_test = read_poses('/home/runyi/Project/TBCP6D/dataset/CambridgeLandmarks_0.5/abs_cambridge_pose_sorted.csv_'+sn+'_test.csv_results.csv')
    camera_poses_train = read_poses('/home/runyi/Project/TBCP6D/dataset/PhotoTourism/'+sn+'_train.csv_results.csv')
    camera_poses_calib = read_poses('/home/runyi/Project/TBCP6D/dataset/PhotoTourism/'+sn+'_val.csv_results.csv')
    camera_poses_test = read_poses('/home/runyi/Project/TBCP6D/dataset/PhotoTourism/'+sn+'_test.csv_results.csv')
    
    camera_poses_train, camera_poses_calib, camera_poses_test = map(normalize_positions, [camera_poses_train, camera_poses_calib, camera_poses_test])


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    frame_length = 0.01
    for pose in camera_poses_train:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame

        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['r', 'r', 'r']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length, normalize=True)

    for pose in camera_poses_calib:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame

        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['g', 'g', 'g']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length, normalize=True)

    for pose in camera_poses_test:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame

        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['b', 'b', 'b']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length, normalize=True)

    # Setting the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('vis/TourismPhoto/Tourism_poses/camera_trans_'+sn+'.png')
