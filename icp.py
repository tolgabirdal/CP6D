import numpy as np
from nc_score import pose_error_transformation
from IPython import embed

def inductive_conformal_prediction(predicted_pose, calibration_data, precomputed_scores, alpha=0.1):
    '''
    predicted_pose: 4x4 transformation matrix, np.array, dtype=np.float32
    calibration_data: list of 4x4 transformation matrix, np.array, dtype=np.float32
    precomputed_scores: list of scores, np.array, dtype=np.float32
    
    delta: p_values
    Pred: prediction region
    '''
    n_cal = len(calibration_data) - 1
    sorted_precomputed_scores = np.sort(precomputed_scores)[::-1]
    
    # Compute test scores & prediction regions
    deltas = []
    for D_j in calibration_data:
        trans, ori = pose_error_transformation(predicted_pose, D_j)
        nc_score = trans + ori
        
        delta = np.mean(sorted_precomputed_scores >= nc_score)
        deltas.append(delta)

    # Compute prediction region
    Pred = np.array([calibration_data[i] for i, delta in enumerate(deltas) if delta >= (1 - alpha)])
    deltas = np.array(deltas)
    # Quantify uncertainties
    # unc = 1 - np.mean(deltas)
    # unct = np.std(deltas)
    return deltas, Pred

def p_value(pred_pose, gt_poses, score, non_conformity_score):
    sorted_non_conformity_score = np.sort(non_conformity_score)[::-1]
    scores = np.zeros(len(gt_poses))
    p_values = np.zeros(len(gt_poses))
    for i in range(len(gt_poses)):
        trans, ori = score(pred_pose, gt_poses[i])
        scores[i] = trans + ori
        count = np.count_nonzero(non_conformity_score >= scores[i])
        p_values[i] = (count + 1) / (len(gt_poses) + 1)
    return p_values

'''
import numpy as np
from nc_score import pose_error_transformation
def inductive_conformal_prediction(predicted_pose, calibration_data, precomputed_scores, alpha=0.1):
    n_cal = len(calibration_data) - 1
    count_valid = 0
    # Compute test scores & delta
    deltas = []
    for index, D_j in enumerate(calibration_data):
        trans, ori = pose_error_transformation(predicted_pose, D_j)
        s_n_plus_1 = trans + ori
        
        delta = np.sum(s_n_plus_1 >= precomputed_scores[:n_cal+1]) / (n_cal + 1)
        deltas.append(delta)
    
    # Compute prediction region
    Pred = np.array([calibration_data[i] for i, delta in enumerate(deltas) if delta >= (1 - alpha)])

    # Quantify uncertainties
    # unc = 1 - np.mean(deltas)
    # unct = np.std(deltas)
    
    return Pred'''