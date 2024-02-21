import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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

sn = 'office'
camera_poses_train = read_poses('./dataset/7Scenes/train_cal_0.5/7Scenes_subset_'+sn+'_train.csv')
camera_poses_calib = read_poses('./dataset/7Scenes/train_cal_0.5/7Scenes_subset_'+sn+'_calib.csv')
camera_poses_test = read_poses('./dataset/7Scenes/train_cal_0.5/abs_7scenes_pose.csv_'+sn+'_test.csv_mstransformer_pred.csv')


# Function to convert quaternion to rotation matrix
def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qxqy, qxqz, qxqw, qyqz, qyqw, qzqw = qx * qy, qx * qz, qx * qw, qy * qz, qy * qw, qz * qw
    
    R = np.array([[1 - 2*(qy2 + qz2), 2*(qxqy - qzqw), 2*(qxqz + qyqw)],
                  [2*(qxqy + qzqw), 1 - 2*(qx2 + qz2), 2*(qyqz - qxqw)],
                  [2*(qxqz - qyqw), 2*(qyqz + qxqw), 1 - 2*(qx2 + qy2)]])
    return R


# Convert quaternions to direction vectors
def quaternion_to_direction_vector(q):
    # Apply quaternion rotation to the Z-axis (0, 0, 1)
    # For simplicity, we'll use a straightforward approach assuming Z-axis orientation
    # Convert quaternion to rotation matrix first
    R = quaternion_to_rotation_matrix(q)
    # Z-axis
    z_axis = np.array([0, 0, 1])
    direction_vector = R @ z_axis
    return direction_vector

# Calculate direction vectors
train_direction_vectors = np.array([quaternion_to_direction_vector(q) for q in camera_poses_train[:, 3:]])
calib_direction_vectors = np.array([quaternion_to_direction_vector(q) for q in camera_poses_calib[:, 3:]])
test_direction_vectors = np.array([quaternion_to_direction_vector(q) for q in camera_poses_test[:, 3:]])

# Plotting
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw a sphere
phi, theta = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

ax.plot_wireframe(x, y, z, color='c', alpha=0.1)

# Plot the points corresponding to the direction vectors
ax.scatter(train_direction_vectors[:,0], train_direction_vectors[:,1], train_direction_vectors[:,2], color='r', s=10)
ax.scatter(calib_direction_vectors[:,0], calib_direction_vectors[:,1], calib_direction_vectors[:,2], color='g', s=10)
ax.scatter(test_direction_vectors[:,0], test_direction_vectors[:,1], test_direction_vectors[:,2], color='b', s=10)


ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig('vis/7scenes_'+sn+'_direction_vectors.png')
