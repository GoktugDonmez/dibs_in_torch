import torch
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

# Fix the imports - use relative imports or sys.path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.dibs_torch_v2 import (
    update_dibs_hparams, bernoulli_soft_gmat, 
    gumbel_acyclic_constr_mc, scores,
    gumbel_soft_gmat, gumbel_grad_acyclic_constr_mc,  # ← Add these missing imports!
    hard_gmat_particles_from_z  # ← Also needed
)


import numpy as np
import logging

# Get a logger for this module
log = logging.getLogger("DiBS_NonLinear")


class NodeFFN(torch.nn.Module):
    """
    One-hidden-layer (or linear) MLP that maps a masked d-dim vector -> scalar.
    Setting hidden=0 gives a single Linear(d,1) so it can reproduce Θ exactly.
    """
    def __init__(self, d: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(d, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 1, bias=True)
            )
        else:                               # shallow == linear surrogate
            self.net = torch.nn.Linear(d, 1, bias=True)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        # x_masked : [..., d]
        return self.net(x_masked).squeeze(-1)  # [...,]


def log_phi_prior(nets: torch.nn.ModuleList,
                  sigma: float = 0.1) -> torch.Tensor:
    """
    Isotropic Gaussian prior over *all* weights & biases in Φ (ModuleList).
    """
    logp = 0.0
    normal = Normal(0.0, sigma)
    for net in nets:
        for p in net.parameters():
            logp += normal.log_prob(p).sum()
    return logp


def log_full_likelihood_mlp(
        data: dict,
        soft_gmat: torch.Tensor,           # [d,d]
        nets: torch.nn.ModuleList,         # list(NodeFFN), length d
        hparams: dict,
        device: str = 'cpu'
    ) -> torch.Tensor:
    """
    log p(Data | Z, Φ)  where Φ is a ModuleList of per-node MLPs.
    """
    x_data = data['x']                                      # [N,d]
    N, d   = x_data.shape
    assert d == soft_gmat.shape[0] == len(nets), "dim mismatch"

    # ------- compute mean column-wise with masking -------------------------
    mu_cols = []
    for j in range(d):
        masked_x = x_data * soft_gmat[:, j]                 # [N,d] ⊙ [d]
        mu_cols.append(nets[j](masked_x))                   # [N]
    pred_mean = torch.stack(mu_cols, dim=1)                 # [N,d]

    # ------- Gaussian log-lik ------------------------------------------------
    sigma_obs = hparams.get('sigma_obs_noise', 0.1)
    log_2pi   = torch.log(torch.tensor(2.0 * torch.pi, device=device))
    residuals = torch.clamp(x_data - pred_mean, -1e3, 1e3)
    log_prob_per_point = -0.5 * (
        log_2pi + 2.0 * torch.log(torch.tensor(sigma_obs, device=device)) +
        (residuals / sigma_obs) ** 2
    )
    log_obs_lik = log_prob_per_point.sum()

    # ------- optional expert edge likelihood (unchanged) -------------------
    log_expert, inv_temp = 0.0, 0.0
    if data.get("y", None):
        inv_temp = hparams.get("temp_ratio", 0.0)
        rho      = hparams["rho"]
        for i, j, val in data["y"]:
            g_ij = soft_gmat[int(i), int(j)]
            p1   = g_ij * (1 - rho) + (1 - g_ij) * rho
            p0   = g_ij * rho       + (1 - g_ij) * (1 - rho)
            log_expert += val * torch.log(p1 + 1e-8) + (1 - val) * torch.log(p0 + 1e-8)

    return log_obs_lik + inv_temp * log_expert


def log_joint_mlp(
        params: dict,                     # needs 'z', 'phi', 't'
        data_dict: dict,
        hparams_cfg: dict,
        device: str = 'cpu'
    ) -> torch.Tensor:
    """
    log P(Z, Φ, Data)  — keeps original acyclicity machinery.
    """
    
    z     = params['z']
    phi   = params['phi']                # ModuleList
    t_val = params.get('t', torch.tensor(0., device=device)).item()

    hparams = update_dibs_hparams(hparams_cfg.copy(), t_val)

    # ---------- likelihood --------------------------------------------------
    g_soft = bernoulli_soft_gmat(z, hparams)
    log_lik = log_full_likelihood_mlp(data_dict, g_soft, phi, hparams, device)

    # ---------- Z-priors (same as before) -----------------------------------
    log_p_z_gauss = torch.distributions.Normal(
        0.0, hparams['sigma_z']).log_prob(z).sum()
    exp_h = gumbel_acyclic_constr_mc(
        z, z.shape[0],
        hparams,
        hparams.get('n_nongrad_mc_samples', hparams['n_grad_mc_samples']),
        device)
    log_p_z_acyc = -hparams['beta'] * exp_h
    log_p_z = log_p_z_gauss + log_p_z_acyc

    # ---------- Φ prior -----------------------------------------------------
    log_p_phi = log_phi_prior(phi, hparams.get('phi_prior_sigma', 1.0))

    return log_lik + log_p_z + log_p_phi


def grad_z_log_joint_gumbel_mlp(current_z_opt, current_phi_nonopt, data_dict, hparams_full, device='cpu'):
    """
    MLP version: Computes gradient of log P(Z, Phi, Data) w.r.t. Z.
    """
    
    d = current_z_opt.shape[0]
    beta = hparams_full['beta']
    sigma_z_sq = hparams_full['sigma_z']**2
    n_grad_mc_samples = hparams_full['n_grad_mc_samples']

    # 1. Gradient of log prior on Z
    n_acyclic_mc_samples = hparams_full.get('n_nongrad_mc_samples', hparams_full['n_grad_mc_samples'])
    grad_expected_h = gumbel_grad_acyclic_constr_mc(
        current_z_opt, d, hparams_full, n_acyclic_mc_samples, create_graph=False
    )
    grad_log_z_prior_acyclicity = -beta * grad_expected_h
    grad_log_z_prior_gaussian = -current_z_opt / sigma_z_sq
    grad_log_z_prior_total = grad_log_z_prior_acyclicity + grad_log_z_prior_gaussian

    log_p_samples_list = []

    for i in range(n_grad_mc_samples):
        g_soft_mc = gumbel_soft_gmat(current_z_opt, hparams_full, device=device)
        
        # Use MLP likelihood
        log_lik_val = log_full_likelihood_mlp(data_dict, g_soft_mc, current_phi_nonopt, hparams_full, device=device)
        
        # Phi prior
        log_phi_prior_val = log_phi_prior(current_phi_nonopt, hparams_full.get('phi_prior_sigma', 1.0))
        
        current_log_density = log_lik_val + log_phi_prior_val
        log_p_samples_list.append(current_log_density)

    log_p_samples = torch.stack(log_p_samples_list, dim=0)
    mean_log_p_samples = log_p_samples.mean()
    grad_curr_log_density_wrt_z, = torch.autograd.grad(outputs=mean_log_p_samples, inputs=current_z_opt)

    # Importance weighting (same as before)
    log_p_max = torch.max(log_p_samples)
    shifted_log_p = log_p_samples - log_p_max
    exp_shifted_log_p = torch.exp(shifted_log_p)

    exp_shifted_log_p_reshaped = exp_shifted_log_p.reshape(-1, *([1]*(current_z_opt.ndim)))
    numerator_sum = torch.sum(exp_shifted_log_p_reshaped * grad_curr_log_density_wrt_z, dim=0)
    denominator_sum = torch.sum(exp_shifted_log_p)

    if denominator_sum < 1e-33:
        grad_log_likelihood_part = torch.zeros_like(current_z_opt)
    else:
        grad_log_likelihood_part = numerator_sum / denominator_sum
        
    return grad_log_z_prior_total + grad_log_likelihood_part


def grad_phi_log_joint(current_z_nonopt, current_phi_opt, data_dict, hparams_full, device='cpu'):
    """
    Computes gradient of log P(Z, Phi, Data) w.r.t. Phi networks.
    Returns the average log density that can be backpropagated.
    """
    
    n_grad_mc_samples = hparams_full['n_grad_mc_samples']

    total_log_density = torch.tensor(0.0, device=device, requires_grad=True)

    for i in range(n_grad_mc_samples):
        g_soft_from_fixed_z = bernoulli_soft_gmat(current_z_nonopt, hparams_full)
        g_hard_sample = torch.bernoulli(g_soft_from_fixed_z)
        
        log_lik_val = log_full_likelihood_mlp(data_dict, g_hard_sample, current_phi_opt, hparams_full, device=device)
        log_phi_prior_val = log_phi_prior(current_phi_opt, hparams_full.get('phi_prior_sigma', 1.0))
        
        current_log_density = log_lik_val + log_phi_prior_val
        total_log_density = total_log_density + current_log_density

    avg_log_density = total_log_density / n_grad_mc_samples
    return avg_log_density  # Return the loss that can be backpropagated


def grad_log_joint_mlp(params, data_dict, hparams_dict_config, device='cpu'):
    """
    MLP version: Computes gradients of the log joint P(Z, Phi, Data) w.r.t Z and Phi.
    Returns both gradients and the loss for backpropagation.
    """
    
    current_z = params['z']
    current_phi = params['phi']
    
    if not current_z.requires_grad:
        current_z.requires_grad_(True)

    t_anneal = params.get('t', torch.tensor([0.0], device=device)).item()
    hparams = update_dibs_hparams(hparams_dict_config.copy(), t_anneal)

    # Compute grad_z using MLP version
    grad_z = grad_z_log_joint_gumbel_mlp(
        current_z_opt=current_z,
        current_phi_nonopt=current_phi,  # Treat phi as fixed for dZ
        data_dict=data_dict,
        hparams_full=hparams,
        device=device
    )
    
    # For phi gradients, we'll use autograd on the joint density
    # This automatically handles the ModuleList gradients
    avg_log_density = grad_phi_log_joint(
        current_z_nonopt=current_z.detach(),  # Treat z as fixed for dPhi
        current_phi_opt=current_phi,
        data_dict=data_dict,
        hparams_full=hparams,
        device=device
    )
    
    return {"z": grad_z, "phi": current_phi, "t": torch.tensor([0.0], device=device)}, avg_log_density


def create_dummy_phi_nets(d=3, hidden_dim=10, seed=None):
    """Creates a dummy ModuleList of NodeFFN networks."""
    if seed is not None:
        torch.manual_seed(seed)
    
    nets = torch.nn.ModuleList()
    for j in range(d):
        net = NodeFFN(d, hidden=hidden_dim)
        nets.append(net)
    
    return nets


def hard_gmat_particles_from_z_mlp(z_particles, alpha_hparam_for_scores=1.0):
    """
    Converts Z particles to hard adjacency matrices by thresholding scores.
    z_particles: [N_particles, D, K, 2]
    alpha_hparam_for_scores: alpha used in scores() function.
    Returns: hard_gmat_particles [N_particles, D, D]
    """
    
    # Get scores for each particle
    s = scores(z_particles, alpha_hparam=alpha_hparam_for_scores) # [N, D, D]
    
    hard_gmats = (s > 0.0).to(torch.float32) # Convert boolean to float (0.0 or 1.0)
    return hard_gmats
