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

sns = ['heads', 'pumpkin', 'redkitchen', 'stairs']
for sn in tqdm(sns):
    camera_poses_train = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_train.csv_results.csv')
    camera_poses_calib = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_cal.csv_results.csv')
    camera_poses_test = read_poses('./dataset/7Scenes_0.5/abs_7scenes_pose.csv_'+sn+'_test.csv_results.csv')


    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for pose in camera_poses_train:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame
        frame_length = 0.15
        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['r', 'r', 'r']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length)

    for pose in camera_poses_calib:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame
        frame_length = 0.15
        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['g', 'g', 'g']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length)

    for pose in camera_poses_test:
        xyz = pose[:3]
        quat = pose[3:]
        R = quaternion_to_rotation_matrix(quat)
        
        # Define a simple coordinate frame
        frame_length = 0.15
        origins = np.array([[0, 0, 0]]).T  # Origin of the frame
        directions = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]) * frame_length  # Axis directions
        transformed_directions = R @ directions  # Rotate the frame
        
        for i, color in zip(range(3), ['b', 'b', 'b']):  # XYZ -> RGB
            ax.quiver(*xyz, *transformed_directions[:, i], color=color, length=frame_length)

    # Setting the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('vis/camera_poses_'+sn+'.png')
