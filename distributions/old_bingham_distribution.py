import numpy as np
from scipy.integrate import tplquad
from scipy.interpolate import Rbf
import functools
from torch import sin, cos

import torch
from tqdm import tqdm
from IPython import embed
from scipy.optimize import minimize
import pandas as pd

# TODO: Use this bingham distribution. https://github.com/igilitschenski/deep_bingham/blob/master/bingham_distribution.py

def bingham_integrand(phi3, phi2, phi1, lambdas):
    """Lambdas should be an array of 4 elements (4th is zero if going by convention) """
    t = np.array([sin(phi1)*sin(phi2)*sin(phi3),
                  sin(phi1)*sin(phi2)*cos(phi3),
                  sin(phi1)*cos(phi2),
                  cos(phi1)])
    exponent = np.sum(lambdas*(t**2))
    return np.exp(exponent)*sin(phi1)**2*sin(phi2)


def create_bingham_interpolator(data_file):
    pass


def bingham_normalization(lambdas):
    """Compute the Bingham normalization of concentration parameters.
    """
    assert len(lambdas) == 4
    f_integrand = functools.partial(bingham_integrand, lambdas=lambdas)
    return tplquad(f_integrand, 0., np.pi,
                   lambda a: 0., lambda a: np.pi,
                   lambda a, b: 0., lambda a, b: 2.*np.pi,
                   epsabs=1e-7, epsrel=1e-3)


def bingham_dist(q, lambdas, coeff_N=None):

    if coeff_N == None:
        coeff_N, _ = bingham_normalization(lambdas)

    return np.exp(np.sum(lambdas*(q**2)))/coeff_N

def bingham_neg_log_likelihood(params, data):
    lambdas, F = params[:4], params[4:].reshape(4, 4)
    log_coeff_N, _ = bingham_normalization(lambdas)
    log_likelihood = 0
    for q in data:
        log_likelihood += np.log(bingham_dist(q, lambdas, log_coeff_N))
    return -log_likelihood
    
def print_progress(xk):
    print("Current iteration parameters:", xk)


def bingham_fit(ground_truth_poses, predicted_poses):
    '''
    Fit a Bingham distribution to the calibration data.
    '''
    # Compute calibration data as the difference between ground truth poses and predicted poses
    calibration_data = ground_truth_poses - predicted_poses

    # Normalize quaternions
    for i in range(calibration_data.shape[0]):
        calibration_data[i] /= np.linalg.norm(calibration_data[i])

    # Initial guess for lambdas and F
    initial_guess = np.zeros(20)
    initial_guess[:4] = np.linspace(0.0, 3.0, 4)
    initial_guess[4:] = np.eye(4).flatten()

    # Optimization with BFGS
    result = minimize(bingham_neg_log_likelihood, initial_guess, args=(calibration_data,), method='BFGS', options={'maxiter': 1000, 'disp': True}, callback=print_progress)

    if result.success:
        optimal_params = result.x
        lambdas_opt, F_opt = optimal_params[:4], optimal_params[4:].reshape(4, 4)
        return lambdas_opt, F_opt
    else:
        print("Optimization failed:", result.message)
        return None

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



def standardize_translation_vectors(translation_vectors):
    # Calculate the mean and standard deviation of the translation vectors
    mean = np.mean(translation_vectors, axis=0)
    std = np.std(translation_vectors, axis=0)

    # Normalize the translation vectors
    normalized_vectors = (translation_vectors - mean) / std

    return normalized_vectors, mean, std



class BinghamDistribution:
    def __init__(self, lambdas=None):
        self.lambdas = lambdas

    @staticmethod
    def bingham_integrand(phi3, phi2, phi1, lambdas):
        t = torch.tensor([np.sin(phi1)*np.sin(phi2)*np.sin(phi3),
                          np.sin(phi1)*np.sin(phi2)*np.cos(phi3),
                          np.sin(phi1)*np.cos(phi2),
                          np.cos(phi1)])
        exponent = torch.sum(lambdas*(t**2))
        return torch.exp(exponent)*np.sin(phi1)**2*np.sin(phi2)

    def bingham_normalization(self, lambdas):
        f_integrand = lambda a, b, c: self.bingham_integrand(a, b, c, lambdas).detach().numpy()
        return tplquad(f_integrand, 0., np.pi,
                       lambda a: 0., lambda a: np.pi,
                       lambda a, b: 0., lambda a, b: 2.*np.pi,
                       epsabs=1e-7, epsrel=1e-3)

    def bingham_dist(self, q, lambdas, coeff_N=None):
        if coeff_N == None:
            coeff_N, _ = self.bingham_normalization(lambdas)
        return torch.exp(torch.sum(lambdas*(q**2)))/coeff_N

    def neg_log_likelihood(self, lambdas, data):
        coeff_N, _ = self.bingham_normalization(lambdas)
        log_likelihood = 0
        for q in data:
            log_likelihood += torch.log(self.bingham_dist(q, lambdas, coeff_N))
        return -log_likelihood

    def fit(self, calibration_data, initial_guess=None, learning_rate=1e-3, num_iterations=1000, tolerance=1e-10):
        if initial_guess is None:
            initial_guess = torch.zeros(4, requires_grad=True)
        else:
            initial_guess = torch.tensor(initial_guess, requires_grad=True)

        calibration_data_torch = torch.tensor(calibration_data)
        optimizer = torch.optim.SGD([initial_guess], lr=learning_rate)

        prev_loss = None

        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            loss = self.neg_log_likelihood(initial_guess, calibration_data_torch)
            loss.backward()
            optimizer.step()

            if prev_loss is not None and abs(prev_loss - loss.item()) < tolerance:
                break

            prev_loss = loss.item()
            tqdm.write("{}:{}, loss:{}".format(i, str(initial_guess), loss))
        self.lambdas = initial_guess.detach()
        return self.lambdas

    def pdf(self, q):
        if self.lambdas is None:
            raise ValueError("Lambdas not set. Call the 'fit' method to estimate lambdas.")
        coeff_N, _ = self.bingham_normalization(self.lambdas)
        return self.bingham_dist(q, self.lambdas, coeff_N)



if __name__ == '__main__':

    # lambdas = np.linspace(0.0, 3.0, 4)
    # coeff_N, err_est = bingham_normalization(lambdas)
    # print('Bingham normalization coefficient: {:}'.format(coeff_N))
    # print('Bingham normalization coefficient error est: {:}'.format(err_est))

    # lambdas_shifted = lambdas + 10.0
    # coeff_N_shifted, err_est_shifted = bingham_normalization(lambdas_shifted)
    # print('Bingham normalization coefficient: {:}'.format(coeff_N_shifted))
    # print('Bingham normalization coefficient error est: {:}'.format(err_est_shifted))


    # # Check invariance
    # q = np.random.rand(4)
    # q = q/np.linalg.norm(q)
    # p1 = bingham_dist(q, lambdas, coeff_N=coeff_N)
    # p2 = bingham_dist(q, lambdas_shifted, coeff_N=coeff_N_shifted)

    # print('Bingham PDF likelihood: {:}'.format(p1))
    # print('Bingham PDF shifted likelihood: {:}'.format(p2))
    
    chess_pred = np.load('/Users/runyi/Project/pyProject/mstransformer/est_data_from_7scenes_mstransformer/est_pose_data_chess.npy')
    chess_gt = read_poses('/Users/runyi/Project/pyProject/mstransformer/datasets/7Scenes/abs_7scenes_pose.csv_chess_test.csv')
    
    # Split Validation Set
    np.random.seed(686)
    rate = 0.5
    idx = np.random.permutation(len(chess_gt))
    calibrate = idx[:int(idx.size * rate)]
    test = idx[int(idx.size * rate):]
    
    # Get Standardised Transition
    cal_chess_gt_trans, cal_chess_gt_mean, cal_chess_gt_std = standardize_translation_vectors(chess_gt[calibrate, 0:3])
    chess_gt_trans = (chess_gt[:, 0:3] - cal_chess_gt_mean) / cal_chess_gt_std
    chess_pred_trans = (chess_pred[:, 0:3] - cal_chess_gt_mean) / cal_chess_gt_std

    # Get Rotation
    chess_gt_q = chess_gt[:, 3:] / np.linalg.norm(chess_gt[:, 3:], axis=1, keepdims=True)
    chess_pred_q = chess_pred[:, 3:] / np.linalg.norm(chess_pred[:, 3:], axis=1, keepdims=True)

    bingham = BinghamDistribution()
    optimal_lambdas = bingham.fit(torch.tensor(chess_gt_q[calibrate[0]]).to(float))
    print("Optimal Lambdas:", optimal_lambdas)