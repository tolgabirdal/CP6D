import logging
import scipy.integrate as integrate
import scipy.optimize
from scipy.optimize import minimize
import scipy.special

import numpy as np
import pandas as pd
import sys
from IPython import embed
from tqdm import tqdm
from numpy import sin, cos

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

class BinghamDistribution:
    def __init__(self, M=None, lambdas=None):
        self.lambdas = lambdas
        self.M = M
        
        self.F_const = self.bingham_normalization(self.lambdas)
        self.dF_const = self.bingham_normalization_derivative(self.lambdas)
        
    def lambdas(self):
        return self.lambdas
    
    def M(self):
        return self.M
    
    def F_const(self):
        return self.F_const
    
    def dF_const(self):
        return self.dF_const
    

    @staticmethod
    def bingham_integrand(phi3, phi2, phi1, lambdas):
        """Lambdas should be an array of 4 elements (4th is zero if going by convention) """
        t = np.array([sin(phi1)*sin(phi2)*sin(phi3),
                    sin(phi1)*sin(phi2)*cos(phi3),
                    sin(phi1)*cos(phi2),
                    cos(phi1)])
        exponent = np.sum(lambdas*(t**2))
        return np.exp(exponent)*sin(phi1)**2*sin(phi2)

    def bingham_normalization(self, lambdas):
        f_integrand = lambda a, b, c: self.bingham_integrand(a, b, c, lambdas)
        F_const, _ =  integrate.tplquad(f_integrand, 
                                            0., 2.0 * np.pi, #phi3
                                            lambda a: 0., lambda a: np.pi, #phi2
                                            lambda a, b: 0., lambda a, b: np.pi, #phi1
                                            epsabs=1e-7, epsrel=1e-3)
        self.F_const = F_const
        return F_const
        
    def bingham_normalization_derivative(self, lambdas):
        derivatives = np.zeros(lambdas.shape)
        def bd_deriv_likelihood(x, j):
            return x[j]**2 * np.exp(np.dot(x, np.dot(np.diag(lambdas), x)))
        for i in range(0, lambdas.shape[0]):
            derivatives[i] = integrate.tplquad(
                lambda phi1, phi2, phi3:
                bd_deriv_likelihood(np.flip(np.array([
                    np.cos(phi1),
                    np.sin(phi1) * np.cos(phi2),
                    np.sin(phi1) * np.sin(phi2) * np.cos(phi3),
                    np.sin(phi1) * np.sin(phi2) * np.sin(phi3),
                ])), i) * (np.sin(phi1) ** 2.) * np.sin(phi2),
                0.0, 2.0 * np.pi,  # phi3
                lambda x: 0.0, lambda x: np.pi,  # phi2
                lambda x, y: 0.0, lambda x, y: np.pi  # phi1
                )[0]
        self.dF_const = derivatives
        return derivatives

    def bingham_covariance(self):
        """Calculate covariance matrix of Bingham distribution.

        Returns
        -------
        s (d x d matrix): scatter/covariance matrix in R^d
        """
        nc_deriv_ratio = np.diag(self.dF_const / self.F_const)

        # The diagonal of D is always 1, however this may not be the
        # case because dF and F are calculated using approximations
        nc_deriv_ratio = nc_deriv_ratio / sum(np.diag(nc_deriv_ratio))

        s = np.dot(self.M,
                   np.dot(nc_deriv_ratio, self.M.transpose()))
        s = (s + s.transpose()) / 2  # enforce symmetry
        return s

    def fit(self, calibration_data):
        '''
        Fit the Bingham distribution to the given data.
        
        Parameters
        ----------
        calibration_data (n x d matrix): n samples of d-dimensional data
        returns: lambdas (d x 1 vector), M (d x d orthogonal matrix): Bingham distribution parameters
        '''

        n_samples = calibration_data.shape[0]
        second_moment = np.dot(calibration_data.T, calibration_data) / n_samples
        return self.fit_second_moment(second_moment, calibration_data)
    
    def fit_second_moment(self, second_moment, calibration_data):
        '''
        Fit the Bingham distribution to the given second moment.
        
        Parameters
        ----------
        second_moment (d x d matrix): second moment of the data
        returns: lambdas (d x 1 vector), M (d x d orthogonal matrix): Bingham distribution parameters
        
        Using MLE estimate for a Bingham Distribution
        '''
        M = self.M
        lambdas = self.lambdas
        F_const = self.F_const
        dF_const = self.dF_const
        
        bd_dim = second_moment.shape[1]
        (moment_eigval, bingham_location) = np.linalg.eig(second_moment)
        
        # Sort eigenvalues (and corresponding eigenvectors) in asc. order.
        eigval_order = np.argsort(moment_eigval)[::-1]
        bingham_location = bingham_location[:, eigval_order]
        moment_eigval = moment_eigval[eigval_order]
        self.M = bingham_location
        # def mle_goal_fun(z, rhs):
        #     """Goal function for MLE optimizer."""

        #     z_param = np.append(z, 0)
        #     norm_const = self.bingham_normalization(z_param)
        #     norm_const_deriv \
        #         = self.bingham_normalization_derivative(z_param)

        #     res = (norm_const_deriv[0:(bd_dim-1)] / norm_const) \
        #         - rhs[0:(bd_dim-1)]
        #     return res

        # bingham_dispersion = scipy.optimize.fsolve(lambda x: mle_goal_fun(x, moment_eigval), np.ones([(bd_dim-1)]))
        # bingham_dispersion = np.append(0, bingham_dispersion)
        def bingham_pdf(q, Z, M, F):
            MZM = np.dot(M, np.dot(np.diag(Z), M.T))
            num_q = np.shape(q)[0]
            density = np.zeros(num_q)
            for i in range(0, num_q):
                density[i] = np.exp(np.dot(q[i].T, np.dot(MZM, q[i])))
            density = density / F
            return density
        def nll(data, F, Z, M):
            return -np.sum(np.log(bingham_pdf(data, Z, M, F)))
        ################
        def objective(lambdas):
            nll_loss = nll(calibration_data, F_const, lambdas, M)
            penalty = 0.0 * np.sum(np.abs(lambdas))
            print(nll_loss, penalty)
            return nll_loss + penalty
        

        bingham_optimal_lambda = minimize(objective, lambdas, method='Nelder-Mead', tol=1e-7)
        self.lambdas = bingham_optimal_lambda.x
        self.F_const = self.bingham_normalization(self.lambdas)
        self.dF_const = self.bingham_normalization_derivative(self.lambdas)
        return self.lambdas, self.M, self.F_const, self.dF_const
        
    def negative_log_likelihood(self, calibration_data):
        # Compute the Bingham PDF
        bingham_pdf = self.pdf(calibration_data)
        # Compute the negative log-likelihood
        nll = -np.sum(np.log(bingham_pdf))
        return nll
    
    def pdf(self, q):
        MZM = np.dot(self.M, np.dot(np.diag(self.lambdas), self.M.T))
        num_q = np.shape(q)[0]
        density = np.zeros(num_q)
        for i in range(0, num_q):
            density[i] = np.exp(np.dot(q[i].T, np.dot(MZM, q[i])))
        density = density / self.F_const
        return density


if __name__ == '__main__':
    # chess_pred = np.load('/Users/runyi/Project/pyProject/mstransformer/est_data_from_7scenes_mstransformer/est_pose_data_chess.npy')
    # chess_gt = read_poses('../7Scenes/abs_7scenes_pose.csv_chess_test.csv')
    
    # # Split Validation Set
    # np.random.seed(686)
    # rate = 0.5
    # idx = np.random.permutation(len(chess_gt))
    # calibrate = idx[:int(idx.size * rate)]
    # test = idx[int(idx.size * rate):]
    
    # # Get Standardised Transition
    # cal_chess_gt_trans, cal_chess_gt_mean, cal_chess_gt_std = standardize_translation_vectors(chess_gt[calibrate, 0:3])
    # chess_gt_trans = (chess_gt[:, 0:3] - cal_chess_gt_mean) / cal_chess_gt_std
    # chess_pred_trans = (chess_pred[:, 0:3] - cal_chess_gt_mean) / cal_chess_gt_std

    # # Get Rotation
    # chess_gt_q = chess_gt[:, 3:] / np.linalg.norm(chess_gt[:, 3:], axis=1, keepdims=True)
    # chess_pred_q = chess_pred[:, 3:] / np.linalg.norm(chess_pred[:, 3:], axis=1, keepdims=True)


    bingham_z = - np.linspace(0.0, 3.0, 4)
    bingham_m = np.eye(4)
    bingham = BinghamDistribution(bingham_m, bingham_z)
    q = np.random.rand(4)
    q = q/np.linalg.norm(q)
    q = np.vstack([q, q])
    bingham.fit(q)
    
    
        
        
    