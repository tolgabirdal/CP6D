import torch

class BinghamDistribution:
    def __init__(self, A, Z):
        self.A = A  # Orthogonal matrix representing principal axes
        self.Z = Z  # Diagonal matrix with concentration parameters

    def pdf(self, x):
        """
        Compute the probability density function for a given point x.
        Note: This function does not include the normalization constant.
        """
        Az = torch.matmul(self.A, self.Z)
        AzA = torch.matmul(Az, self.A.t())
        return torch.exp(torch.matmul(torch.matmul(x.t(), AzA), x))

    # Sampling method and normalization constant calculation
    # would be implemented here. These are non-trivial and require
    # specialized algorithms.

# Example usage
A = torch.eye(4)  # Example orthogonal matrix
Z = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0]))  # Example concentration parameters

bingham_dist = BinghamDistribution(A, Z)

# Example point on the unit sphere
x = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)

# Compute the PDF at point x
pdf_value = bingham_dist.pdf(x)
print(pdf_value)