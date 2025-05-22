import torch

from models.dibs_torch_v2 import log_joint, grad_log_joint, bernoulli_soft_gmat, hard_gmat_particles_from_z, log_gaussian_likelihood
from models.graph_torch import scalefree_dag_gmat
from models.utils_torch import sample_x

def main():
    print("Hello, World!")

    n = 10
    d = 5
    x = torch.randn(n, d)
    pred_mean = torch.randn(n, d)
    sigma = torch.rand(d)

    print(log_gaussian_likelihood(x, pred_mean, sigma))

if __name__ == "__main__":
    main()

