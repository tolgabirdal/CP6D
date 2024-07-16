import numpy as np
import torch
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from icp import Inductive_Conformal_Predcition as ICP
from distributions.gaussian_distribution import GaussianUncertainty
from distributions.bingham_distribution import BinghamDistribution
from nc_score import pose_err
from tqdm import tqdm
import tools
# import rpmg
import argparse
from dataset.CameraPoseDataset import CameraPoseDatasetPred
from skimage.io import imread
import cv2
# from Keypoint.ALIKED import aliked_kpts
import torch.nn.functional as F
import torch_bingham
from IPython import embed
from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_write_dense import read_array

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


def load_npz_file(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    img_paths = data['img_path']
    feature_t = data['feature_t']
    feature_rot = data['feature_rot']
    return img_paths, feature_t, feature_rot

def calculate_cosine_similarity_pytorch(feature, dataset_features):
    # Ensure inputs are torch tensors
    
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

def find_and_match_features(image1, image2, model):
    # Create SIFT object
    # if model == 'sift':
    #     sift = cv2.SIFT_create()
    # elif model == 'aliked-n32':
    #     keypoint_detector = aliked_kpts.model_selection('aliked-n32',top_k=1000, device=device)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    # kp1, des1 = aliked_kpts.keypoint(image1, model)
    # kp2, des2 = aliked_kpts.keypoint(image2, model)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append(m)
    return kp1, kp2, good_matches

# Function to estimate essential matrix and recover pose
def estimate_pose(kp1, kp2, matches, K):
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask = cv2.recoverPose(E, points1, points2, K)
    return R, t

def find_poses(image1, image2, model, K):
    kp1, kp2, good_matches = find_and_match_features(image1, image2, model)
    # K = np.array([[948, 0, 960],
    #               [0, 533, 540],
    #               [0, 0, 1]], dtype=np.float32)
    # Cambridge 
    
    # 7Scenes
    # K = np.array([[532.57, 0, 320],
    #               [0, 531.54, 240],
    #               [0, 0, 1]], dtype=np.float32)
    
    R, t = estimate_pose(kp1, kp2, good_matches, K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.squeeze()  # Ensure t is correctly shaped
    return T


def rotation_err(est_pose, gt_pose):
    """
    Calculate the orientation error given the estimated and ground truth pose(s).
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: orientation error(s)
    """
    est_pose_q = F.normalize(est_pose, p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose, p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))

    if torch.abs(inner_prod) <= 1:
        orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / torch.pi
    else:
        origin = torch.abs(torch.abs(inner_prod) - int(torch.abs(inner_prod)) - 1)
        orient_err = 2 * torch.acos(origin) * 180 / torch.pi
    return orient_err

def translation_err(est_pose, gt_pose):
    """
    Calculate the position error given the estimated and ground truth pose(s).
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    return posit_err

def compute_covariance_matrix(quaternions):
    # Subtract mean
    mean_quaternion = np.mean(quaternions, axis=0)
    centered_quaternions = quaternions - mean_quaternion
    # Compute covariance matrix
    covariance_matrix = np.dot(centered_quaternions.T, centered_quaternions) / len(quaternions)
    return covariance_matrix

def fit_bingham_distribution(quaternions):
    covariance_matrix = compute_covariance_matrix(quaternions)
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Compute concentration parameters
    lambdas = 1 / eigenvalues - 1 / eigenvalues[-1]
    lambdas -= lambdas.min()
    return eigenvectors, lambdas

def get_pred_region(icp, test_data_loader, Un, norm):
    pred_regions = []
    origin_trans_errs = []
    origin_rot_errs = []
    test_gt_poses = []
    test_pred_poses = []
    uncertainties = []
    for i, minibatch in enumerate(tqdm(test_data_loader)):
        # test_img = minibatch['img'].squeeze(0).detach().numpy()
        test_t_gt = minibatch['pose'][:, :3]
        test_t = minibatch['est_pose'][:, :3]
        test_q_gt = minibatch['pose'][:, 3:]
        test_q = minibatch['est_pose'][:, 3:]
        
        test_gt_poses.append(minibatch['pose'])
        test_pred_poses.append(minibatch['est_pose'])
        # test_R = compute_rotation_matrix_from_quaternion(test_q, n_flag=True).squeeze()
        test_pose = minibatch['est_pose']
        pred_region_idx_cal = icp.compute_p_value_from_calibration_poses(test_pose, topk=5)

        # print(pred_region_idx_cal)
        pred_region = cal_poses[pred_region_idx_cal]
        try:
            uncertainty = Un.compute_uncertainty_score_entropy(pred_region)
        except:
            print(minibatch['img_path'])
            uncertainty = uncertainties[-1]
        uncertainties.append(uncertainty)
        pred_regions.append(pred_region)
        # origin_trans_errs.append(translation_err(test_t * (norm[1] - norm[0]), test_t_gt * (norm[1] - norm[0])))
        origin_trans_errs.append(translation_err(test_t , test_t_gt ))
        origin_rot_errs.append(rotation_err(test_q, test_q_gt))

    return {
        'test_gt': test_gt_poses,
        'test_pred': test_pred_poses,
        'pred_regions': pred_regions,
        'Trans_Err': torch.tensor(origin_trans_errs),
        'Rot_Err': torch.tensor(origin_rot_errs),
        'uncertainties': torch.tensor(uncertainties)
    }
    
def draw_data(args, ori_err, new_err, uncertainty_set, mode='Translation'):
    plt.figure(figsize=(10, 10))
    plt.plot(uncertainty_set, new_err, 'o-', color='b', label='Conformal '+mode+' Error')
    plt.axhline(y=ori_err.mean(), color='g', linestyle='--', label='Original '+mode+' Error')
    plt.xlabel('Uncertainty Level')
    plt.ylabel('Mean '+mode+' Error')
    plt.title('Mean '+mode+' Error')
    for i, txt in enumerate(new_err):
        plt.annotate(f'{txt:.3f}', (uncertainty_sets[i], new_err[i]), textcoords="offset points", xytext=(0,3), ha='center')
        
    plt.annotate(f'{ori_err.mean():.3f}', xy=(0.1, ori_err.mean()), textcoords="offset points", xytext=(0,3), ha='right', color='g')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/home/runyi/Project/TBCP6D/experiments/visualization/vis_conformal_t/'+args.data+'/'+args.sn+'_'+args.exp+'.png')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-r", "--root_path", help="dataset root path, e.g. /home/runyi/Data/")
    arg_parser.add_argument("-d", "--data", help="dataset, e.g. 7Scenes, PhotoTourism, CambridgeLandmarks")
    arg_parser.add_argument("-l", "--label_file", help="label files dir, /home/runyi/Project/TBCP6D/dataset/PhotoTourism/")
    arg_parser.add_argument("-s", "--sn", help="name of scenes e.g. chess, fire")
    arg_parser.add_argument("-f", "--feature", help="if you need feature")
    arg_parser.add_argument("--exp", default=None, help="name of experiment")
    args = arg_parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.data == "7Scenes":
        args.sn = 'abs_7scenes_pose.csv_' + args.sn
        final_cal = '_cal.csv_results.csv'
        final_test = '_test.csv_results.csv'
    elif args.data == "CambridgeLandmarks":
        args.sn = 'abs_cambridge_pose_sorted.csv_' + args.sn
        final_cal = '_cal.csv_results.csv'
        final_test = '_test.csv_results.csv'
    elif args.data == "PhotoTourism":
        final_cal = '_val.csv_results.csv'
        final_test = '_test.csv_results.csv'
        
    cal_labels_file = args.label_file + args.sn + final_cal
    test_labels_file = args.label_file + args.sn + final_test
    # if args.feature is not None:
    #     calibration_img_path, calibration_feature_t, calibration_feature_rot = load_npz_file(args.label_file+args.sn+'_val.csv_results.npz')
    #     calibration_feature_t, calibration_feature_rot = torch.tensor(calibration_feature_t), torch.tensor(calibration_feature_rot)

    data_path = args.root_path + args.data + '/'
    cal_set = CameraPoseDatasetPred(data_path, cal_labels_file, load_npz=False)
    test_set = CameraPoseDatasetPred(data_path, test_labels_file, load_npz=False)
    cal_poses = torch.tensor(cal_set.poses)
    cal_pred_poses = torch.tensor(cal_set.pred_poses)
    tmean, tstd, tmax, tmin = torch.mean(cal_poses[:, :3], dim=0), torch.std(cal_poses[:, :3], dim=0), torch.max(cal_poses[:, :3], dim=0)[0], torch.min(cal_poses[:, :3], dim=0)[0]
    # cal_poses[:, :3] = (cal_poses[:, :3] - tmin) / (tmax - tmin)
    # cal_pred_poses[:, :3] = (cal_pred_poses[:, :3] - tmin) / (tmax - tmin)
    trans_norm = [tmax, tmin]
    # test_set.poses[:, :3] = (test_set.poses[:, :3] - np.array(tmin)) / (np.array(tmax) - np.array(tmin))
    # test_set.pred_poses[:, :3] = (test_set.pred_poses[:, :3] - np.array(tmin)) / (np.array(tmax) - np.array(tmin))

    # calib non-conformity
    icp = ICP(cal_poses, cal_pred_poses, mode='Trans')
    bingham_z = - np.linspace(0.0, 3.0, 4)[::-1]
    bingham_m = np.eye(4)
    BU = BinghamDistribution(bingham_m, bingham_z, {"norm_const_mode": "numerical"})
    GU = GaussianUncertainty(args.data)

    dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    
    
    uncertainty_sets = np.linspace(0.02, 1.0, 50)
    
    mean_rot_err = []
    random_prune_rot_err = []
    mean_t_err = []
    random_prune_t_err = []
    num_effiect_samples = []
    
    pred_data = get_pred_region(icp, dataloader, GU, (tmax, tmin))
    # embed()
    # embed()
    valid_uncertainties = pred_data['uncertainties'][~torch.isnan(pred_data['uncertainties'])]
    
    pred_data['uncertainties'] = (pred_data['uncertainties'] - valid_uncertainties.min()) / (valid_uncertainties.max() - valid_uncertainties.min())
    ori_t_err = pred_data['Trans_Err'] 
    ori_r_err = pred_data['Rot_Err']
    embed()
    for uncertainty_set in uncertainty_sets:
        total_count = int(len(pred_data['uncertainties']) * uncertainty_set) - 1
        threshold = torch.sort(pred_data['uncertainties'])[0][total_count]
        mask = (pred_data['uncertainties'] <= threshold)
        # (err_trans * mask_trans).sum().item() / (mask_trans.sum().item() + 1e-9)
        mean_t_err.append((pred_data['Trans_Err'] * mask).sum() / (mask.sum() + 1e-9))
        mean_rot_err.append((pred_data['Rot_Err'] * mask).sum() / (mask.sum() + 1e-9))
        num_effiect_samples.append(mask.sum())

    draw_data(args, ori_t_err, mean_t_err, uncertainty_sets, mode='Translation')
    
    
    #     p_values_rot = []
    #     p_values_t = []


    # embed()
    #     print("Uncertainty Set: ", uncertainty_set, len(new_t_err))
    #     num_effiect_samples.append(len(new_t_err))
    #     np.random.seed(42)
    #     ori_random_t_err = np.random.choice(ori_t_err, size=int(len(new_t_err)), replace=False)
    #     mean_t_err.append(np.mean(new_t_err))
    #     random_prune_t_err.append(np.mean(ori_random_t_err))
    #     print("Uncertainty Set: ", uncertainty_set, "Mean Translation Error: ", np.mean(new_t_err), "Random Prune Translation Error: ", np.mean(ori_random_t_err), "Original Translation Error: ", np.mean(ori_t_err), "Total: ", len(ori_t_err))

    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 1, 1)
    # plt.title(args.data + ": " + args.sn)
    # plt.plot(uncertainty_sets, mean_t_err, 'o-', color='b', label='Conformal Translation Error')
    # # plt.plot(uncertainty_sets, random_prune_t_err, 'x-', color='r', label='Random Prune Translation Error')
    # plt.axhline(y=ori_t_err.mean(), color='g', linestyle='--', label='Original Translation Error')
    # plt.xlabel('Uncertainty Level')
    # plt.ylabel('Mean Translation Error')
    # for i, txt in enumerate(mean_t_err):
    #     plt.annotate(f'{txt:.3f}', (uncertainty_sets[i], mean_t_err[i]), textcoords="offset points", xytext=(0,3), ha='center')
        
    # plt.annotate(f'{ori_t_err.mean():.3f}', xy=(0.1, ori_t_err.mean()), textcoords="offset points", xytext=(0,3), ha='right', color='g')
    # # plt.title('Mean Translation Error')
    # # plt.legend()
    # # plt.tight_layout()
    # # plt.savefig('vis/TourismPhoto/real_conformal_t/'+args.sn+'_mean_t_err.png')

    # plt.subplot(2, 1, 2)
    # # Plot the length of new_t_err
    # plt.plot(uncertainty_sets, num_effiect_samples, 'o-', color='m', label='Length of Valid Poses')
    # plt.axhline(y=len(ori_t_err), color='g', linestyle='--', label='Total Samples')
    # # Add labels and title
    # plt.xlabel('Uncertainty Level')
    # plt.ylabel('Num of Valid Predictions')
    # for i, txt in enumerate(num_effiect_samples):
    #     plt.annotate(f'{txt}', (uncertainty_sets[i], num_effiect_samples[i]), textcoords="offset points", xytext=(0,3), ha='center')
    # plt.annotate(f'{len(ori_t_err)}', xy=(0.1, len(ori_t_err)), textcoords="offset points", xytext=(0,3), ha='right', color='g')
    # # plt.title('Length of new_t_err')
    # # Add legend
    # plt.legend()
    # # Adjust the layout
    # plt.tight_layout()
    # # Save the figure
    # plt.savefig('vis_conformal_r/'+ args.data + '/' + args.sn+ '_' + args.exp + '.png')