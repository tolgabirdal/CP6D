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
from dataset.CameraPoseDataset import CameraPoseDatasetPred
from skimage.io import imread
from IPython import embed

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

def compute_relative_pose(pose1, pose2):
    '''
    pose1 and pose2 are nx7 tensors, where the first 3 elements are the translation and the last 4 elements are the quaternion.
    '''
    # Assuming `tools.compute_rotation_matrix_from_quaternion` is correctly defined elsewhere
    R1 = compute_rotation_matrix_from_quaternion(pose1[:, 3:])
    R2 = compute_rotation_matrix_from_quaternion(pose2[:, 3:])
    t1 = pose1[:, :3].unsqueeze(-1)  # Ensure t1 is nx3x1 for correct broadcasting in matrix operations
    t2 = pose2[:, :3].unsqueeze(-1)  # Ensure t2 is nx3x1
    # Compute the relative rotation
    relative_R = R2.bmm(R1.transpose(1, 2))
    
    # Compute the relative translation
    relative_t = t2 - relative_R.bmm(t1)

    # Flatten relative_t back to nx3 for concatenation
    relative_t = relative_t.squeeze(-1)

    # For returning, you may want to convert relative_R back to quaternions and concatenate with relative_t
    # Assuming `tools.compute_quaternion_from_rotation_matrix` is a function that converts rotation matrices to quaternions
    relative_quaternions = compute_quaternions_from_rotation_matrices(relative_R)
    relative_pose = torch.cat((relative_t, relative_quaternions), dim=1)

    return relative_pose

def compute_and_compare_pose(T_r, test_pose, gt_pose):
    """
    Compute the final pose using the relative pose T_r and test_pose, 
    and compare it with the ground truth pose (gt_pose).
    
    :param T_r: numpy array of shape (4, 4), the relative transformation matrix
    :param test_pose: torch.Tensor of shape (1, 7), the test pose in translation + quaternion format
    :param gt_pose: numpy array of shape (7,), the ground truth pose in translation + quaternion format
    :return: position error and orientation error
    """
    # Convert quaternions to rotation matrices
    R_test = compute_rotation_matrix_from_quaternion(test_pose[:, 3:], n_flag=True)

    # Create transformation matrices for test_pose
    T_test = torch.zeros((4, 4))
    T_test[:3, :3] = R_test.squeeze(0)
    T_test[:3, 3] = test_pose[:, :3].squeeze(0)
    T_test[3, 3] = 1
    
    # Convert numpy T_r to tensor and apply relative transformation
    T_final = torch.mm(torch.FloatTensor(T_r), T_test)

    # Convert the final transformation matrix back to translation and quaternion format
    t_final = T_final[:3, 3]
    R_final = T_final[:3, :3]
    quaternion_final = compute_quaternions_from_rotation_matrices(R_final.unsqueeze(0))

    # Final pose in translation + quaternion format
    final_pose = torch.cat((t_final, quaternion_final.squeeze(0)), 0).unsqueeze(0)

    # Compare with gt_pose
    gt_pose_tensor = torch.FloatTensor(gt_pose).unsqueeze(0)
    posit_err, orient_err = pose_err(final_pose, gt_pose_tensor)

    return posit_err.item(), orient_err.item()

def load_npz_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    img_paths = data['img_path']
    feature_t = data['feature_t']
    feature_rot = data['feature_rot']
    return img_paths, feature_t, feature_rot

def calculate_cosine_similarity_pytorch(feature, dataset_features):
    # Ensure inputs are torch tensors
    feature = torch.tensor(feature).float().unsqueeze(0)  # Convert to 2D tensor for CosineSimilarity
    dataset_features = torch.tensor(dataset_features).float()
    
    # Initialize CosineSimilarity object
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    
    # Calculate cosine similarity
    # CosineSimilarity expects inputs of the same dimension, so we broadcast feature to match dataset_features
    similarities = cos_sim(feature, dataset_features)
    
    return similarities.numpy()  # Convert back to numpy array for compatibility

def find_most_similar_image(new_image_feature, dataset_features, img_paths):
    # Calculate cosine similarity for each feature in the dataset using PyTorch
    similarities = calculate_cosine_similarity_pytorch(new_image_feature, dataset_features)
    
    # Find the index of the highest similarity score
    most_similar_index = np.argmax(similarities)
    
    # Retrieve the path for the most similar image
    most_similar_image_path = img_paths[most_similar_index]
    
    return most_similar_image_path

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--sn", help="name of scenes e.g. chess, fire")
    args = arg_parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = '/home/runyi/Data/7Scenes'
    cal_labels_file = '/home/runyi/Project/TBCP6D/dataset/7Scenes_0.5/abs_7scenes_pose.csv_chess_cal.csv_results.csv'
    test_labels_file = '/home/runyi/Project/TBCP6D/dataset/7Scenes_0.5/abs_7scenes_pose.csv_chess_test.csv_results.csv'
    cal_set = CameraPoseDatasetPred(dataset_path, cal_labels_file)
    test_set = CameraPoseDatasetPred(dataset_path, test_labels_file)
    
    calibration_img_path, calibration_feature_t, calibration_feature_rot = load_npz_file('/home/runyi/Project/TBCP6D/dataset/7Scenes_0.5/abs_7scenes_pose.csv_chess_cal.csv_results.csv_results.npz')

    calibration_feature_t, calibration_feature_rot = torch.tensor(calibration_feature_t), torch.tensor(calibration_feature_rot)

    # calib non-conformity
    icp_rot = ICP_ROT(torch.tensor(cal_set.poses[:, 3:]), torch.tensor(cal_set.pred_poses[:, 3:]))
    calib_rot_nc = icp_rot.compute_non_conformity_scores()
    
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    for i, minibatch in enumerate(dataloader):
        for k, v in minibatch.items():
            minibatch[k] = v.to(device)
            
            test_feature_rot = minibatch['feature_rot']
            embed()
            
            
            
        