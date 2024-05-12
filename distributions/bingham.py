import torch
import torch_bingham
from torch.linalg import eigh
import numpy as np
# from scipy.linalg import eigh
from scipy.optimize import minimize
from IPython import embed

class BinghamUncertainty():
    def __init__(self) -> None:
        super(BinghamUncertainty, self).__init__()
        pass

    def fit(self, X):
        """Fit a Bingham distribution to data X where X is (n, d) and d=3 for quaternions."""
        n, d = X.shape

        # Calculate the scatter matrix
        S = torch.mm(X.t(), X) / n

        # Perform eigenvalue decomposition
        evals, V = eigh(S)

        # Use scipy to optimize Z
        initial_Z = -torch.sort(-evals)[0][:d-1].cpu().numpy()
        embed()

        def cost(Z):
            F, dF = self.bingham_F(torch.from_numpy(Z).float().unsqueeze(0))

            # We convert F and dF back to numpy to compute the cost
            F = F.item()
            dF = dF.cpu().numpy()
            
            return np.sum((dF / F - evals[:d-1].cpu().numpy()) ** 2)

        res = minimize(cost, initial_Z, method='BFGS')
        self.Z = torch.tensor(res.x, dtype=torch.float32).unsqueeze(0)

        self.M = V[:, :d-1]

    def bingham_F(self, Z):
        # Placeholder: Implement the computation of F and dF based on self.Z
        # For example, let's return dummy values
        F = torch_bingham.F_lookup_3d(Z)
        dF = torch_bingham.dF_lookup_3d(Z)  # Not correct, just as a placeholder
        return F, dF

if __name__ == '__main__':
    # Example usage
    bu = BinghamUncertainty()
    q = torch.tensor([[-3.6923,  0.3001,  2.9537,  0.8756,  0.0681,  0.4641, -0.1157],
        [-3.8679,  0.7375,  4.2710,  0.8774,  0.0815,  0.4587, -0.1143],
        [-3.6118,  0.4946,  2.8798,  0.8835,  0.0672,  0.4443, -0.1324],
        [-3.8952,  0.2775,  2.5234,  0.8787,  0.0845,  0.4543, -0.1198],
        [-3.7988,  0.3327,  3.0719,  0.8891,  0.0787,  0.4363, -0.1137]],
       dtype=torch.float64)
    bu.fit(q[:, 3:])
    print("M:", bu.M)
    print("Z:", bu.Z)
    
