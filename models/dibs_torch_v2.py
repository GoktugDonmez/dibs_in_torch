import torch
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
import numpy as np
import logging

# Get a logger for this module
log = logging.getLogger("DiBS")

# --- Assumed utility functions (would need to be translated from models/utils.py) ---
def acyclic_constr_old(g_mat, d):
    """
    Computes the acyclicity constraint h(G) = tr((I + alpha*G)^d) - d.
    g_mat: [d, d] tensor, the graph adjacency matrix (can be soft)
    d: number of nodes
    """
    alpha = 1.0 / d
    eye = torch.eye(d, device=g_mat.device, dtype=g_mat.dtype)
    m = eye + alpha * g_mat
    # Efficiently compute matrix power for integer d
    # For non-integer d, or for very large d where this is slow,
    # other methods like repeated squaring or eigendecomposition might be needed
    # if d is small, torch.linalg.matrix_power is fine.
    # For larger d, this can be computationally intensive.
    # The original DiBS paper uses d in the exponent.
    m_mult = torch.linalg.matrix_power(m, d)


    h = torch.trace(m_mult) - d
    return h
def acyclic_constr(g_mat, d):
    """Efficient fixed acyclicity constraint using eigendecomposition or series expansion."""
    alpha = 1.0 / d
    eye = torch.eye(d, device=g_mat.device, dtype=g_mat.dtype)
    m = eye + alpha * g_mat
    
    # For small d, use matrix power
    if d <= 10:
        m_mult = torch.linalg.matrix_power(m, d)
        return torch.trace(m_mult) - d
    
    # For larger d, use eigendecomposition or series approximation
    try:
        eigenvals = torch.linalg.eigvals(m)
        # trace(M^d) = sum(lambda_i^d)
        h = torch.sum(torch.real(eigenvals ** d)) - d
        return h
    except:
        # Fallback to series expansion for very large or ill-conditioned matrices
        # h(G) ≈ trace(sum_{k=1}^{d} (alphaG)^k) using series expansion
        trace_sum = torch.tensor(0.0, device=g_mat.device, dtype=g_mat.dtype)
        power_g = g_mat
        for k in range(1, min(d+1, 20)):  # Truncate series for efficiency
            trace_sum += (alpha ** k) * torch.trace(power_g) / k
            if k < min(d, 19):
                power_g = torch.matmul(power_g, g_mat)
        return trace_sum

def stable_mean(fxs, dim=0, keepdim=False):
    """
    Computes a stable mean based on the JAX implementation logic provided.
    This version handles positive and negative values separately using logsumexp
    for potentially improved stability with numbers of widely varying magnitudes.

    Args:
        fxs (torch.Tensor): Input tensor.
        dim (int or tuple of ints, optional): The dimension or dimensions to reduce.
            If None (default in underlying logic if input `dim` leads to it, e.g. for scalar),
            reduces all dimensions. Default is 0.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not.
            Default is False.

    Returns:
        torch.Tensor: Tensor with the stable mean.

    Note:
    - For empty inputs or slices (e.g., a dimension of size 0), this
      implementation returns 0. This differs from `torch.mean`, which returns NaN.
    - If `fxs` is a scalar and `dim` is specified (e.g., `dim=0`), this function
      treats `dim` as `None` (reducing all dimensions, i.e., returning the scalar
      itself). `torch.mean` would raise an error in such a case.
    - The input tensor `fxs` will be converted to a floating-point dtype if it's
      not already, as `torch.log` requires float or complex inputs.
    """
    jitter = 1e-30

    if not fxs.is_floating_point():
        # Promote fxs to the default floating-point type for log operations
        fxs = fxs.to(torch.get_default_dtype())

    # Determine effective dim and keepdim for internal operations
    # If fxs is scalar, dim effectively becomes None, and keepdim is effectively False
    # as the output will be scalar.
    is_scalar_input = (fxs.ndim == 0)
    effective_dim = dim
    if is_scalar_input:
        effective_dim = None

    # Configure reduction dimensions and total element count for the mean
    if effective_dim is None:
        # Reduce over all dimensions
        sum_reduce_dims = list(range(fxs.ndim)) if fxs.ndim > 0 else []
        n_total_elements_val = fxs.numel()
        # When reducing all dims, internal keepdim for logsumexp/sum should be False
        # as the intermediate sum (sum_psve, sum_ngve_abs) will be scalar.
        internal_keepdim = False
    else:
        # Reduce over specified dimension(s)
        sum_reduce_dims = effective_dim
        if isinstance(effective_dim, int):
            n_total_elements_val = fxs.shape[effective_dim]
        else:  # effective_dim is a tuple of dimensions
            n_total_elements_val = 1
            for d_idx in effective_dim:
                n_total_elements_val *= fxs.shape[d_idx]
        # Internal sums should keep the reduced dimension for broadcasting
        internal_keepdim = True

    n_total = torch.tensor(n_total_elements_val, device=fxs.device, dtype=fxs.dtype)

    # Create masks for positive and negative values
    positive_mask = fxs > 0.0
    negative_mask = fxs < 0.0

    # Isolate contributions from positive and absolute negative values
    # Values are original where mask is true, 0 otherwise.
    fxs_psve_contrib = fxs * positive_mask
    fxs_ngve_contrib = -fxs * negative_mask  # Absolute values of negative contributions

    # Calculate sum of positive contributions using log-sum-exp
    # torch.log(0.0) is -inf. torch.logsumexp handles -inf correctly.
    # If all contributions are 0 (no positive values), sum will be exp(-inf) = 0.
    log_fxs_psve_contrib = torch.log(fxs_psve_contrib)
    sum_psve = torch.exp(torch.logsumexp(log_fxs_psve_contrib, dim=sum_reduce_dims, keepdim=internal_keepdim))

    # Calculate sum of absolute negative contributions using log-sum-exp
    log_fxs_ngve_contrib = torch.log(fxs_ngve_contrib)
    sum_ngve_abs = torch.exp(torch.logsumexp(log_fxs_ngve_contrib, dim=sum_reduce_dims, keepdim=internal_keepdim))

    # Count positive elements and non-positive elements (JAX's n_ngve logic)
    # Ensure counts match fxs.dtype and dimensionality of sums for broadcasting.
    n_psve = torch.sum(positive_mask, dim=sum_reduce_dims, keepdim=internal_keepdim).to(fxs.dtype)

    if effective_dim is None: # n_total is scalar, n_psve is scalar
        n_ngve_jax = n_total - n_psve
    else: # n_total is scalar (shape[dim]), n_psve is a tensor due to internal_keepdim=True
        n_ngve_jax = n_total.to(fxs.dtype) - n_psve # n_total broadcasts

    # Calculate average of positive contributions
    # Initialize with zeros; shape matches sum_psve
    avg_psve = torch.zeros_like(sum_psve)
    # Mask for elements where n_psve > 0 to avoid division by zero
    psve_calc_mask = n_psve > 0
    # n_psve and sum_psve have same shape due to consistent internal_keepdim.
    # So direct indexing with psve_calc_mask is fine.
    if torch.any(psve_calc_mask): # Avoids indexing empty tensor if all n_psve are 0
        avg_psve[psve_calc_mask] = sum_psve[psve_calc_mask] / (n_psve[psve_calc_mask] + jitter)

    # Calculate average of absolute negative contributions (JAX's avg_ngve logic)
    avg_ngve = torch.zeros_like(sum_ngve_abs)
    # Mask for elements where n_ngve_jax > 0 (denominator for this average)
    ngve_calc_mask = n_ngve_jax > 0
    if torch.any(ngve_calc_mask):
        avg_ngve[ngve_calc_mask] = sum_ngve_abs[ngve_calc_mask] / (n_ngve_jax[ngve_calc_mask] + jitter)

    # Combine terms based on the JAX formula structure
    # n_total is scalar. Add jitter for safety if n_total_elements_val could be 0.
    safe_n_total = n_total + jitter
    # If n_total_elements_val is 0, safe_n_total is jitter. All sums and counts
    # (sum_psve, sum_ngve_abs, n_psve, n_ngve_jax) will be 0. Result becomes 0.

    term_psve = (n_psve / safe_n_total) * avg_psve
    term_ngve = (n_ngve_jax / safe_n_total) * avg_ngve
    result = term_psve - term_ngve

    # Final keepdim handling
    if is_scalar_input:
        # For scalar input, result is already scalar. keepdim is effectively False.
        return result
    
    if effective_dim is not None and not keepdim:
        # Squeeze the reduced dimension(s) if keepdim is False
        result = result.squeeze(effective_dim)
    
    return result

def expand_by(arr, n):
    """
    Expands the tensor by n dimensions at the end.
    arr: input tensor
    n: number of dimensions to add
    """
    for _ in range(n):
        arr = arr.unsqueeze(-1)
    return arr
# --- End of assumed utility functions ---


# === Core DiBS Logic in PyTorch ===

# Function Calling Structure:
# SVGD loop (not shown here, but would be in `fit_svgd`) calls:
#   `grad_log_joint` (to get gradients for Z and Theta)
#     `grad_log_joint` calls:
#       `update_dibs_hparams` (to anneal hyperparameters)
#       `grad_z_log_joint_gumbel` (for dZ)
#         `grad_z_log_joint_gumbel` calls:
#           `gumbel_grad_acyclic_constr_mc` (for prior gradient part)
#             `gumbel_grad_acyclic_constr_mc` calls (conceptually, via torch.autograd):
#               `gumbel_soft_gmat` (to get G_soft from Z)
#               `acyclic_constr` (the constraint function)
#           `log_full_likelihood` (for likelihood gradient part, via torch.autograd)
#             `log_full_likelihood` calls:
#               `log_gaussian_likelihood`
#               `log_bernoulli_likelihood` (if expert data `y` is present)
#           `log_theta_prior` (for likelihood gradient part, via torch.autograd)
#           `gumbel_soft_gmat` (multiple times for MC estimates)
#       `grad_theta_log_joint` (for dTheta)
#         `grad_theta_log_joint` calls:
#           `log_full_likelihood` (via torch.autograd)
#           `log_theta_prior` (via torch.autograd)
#           `bernoulli_soft_gmat` (multiple times for MC estimates)
#
# The `log_joint` function itself (calculating the value, not gradient) calls:
#   `update_dibs_hparams`
#   `log_full_likelihood`
#   `gumbel_acyclic_constr_mc` (or `bernoulli_soft_gmat` depending on formulation)
#   `log_theta_prior`

def log_gaussian_likelihood(x, pred_mean, sigma=0.1):
    """
    More numerically stable version of Gaussian log likelihood.
    """
    # Ensure sigma is reasonable
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(sigma, dtype=pred_mean.dtype, device=pred_mean.device)
    
    sigma = torch.clamp(sigma, min=1e-12, max=1e5)  # Prevent extreme values
    
    # Compute residuals
    residuals = x - pred_mean
    
    # Clamp residuals to prevent overflow in squared term
    residuals = torch.clamp(residuals, min=-1e3, max=1e3)
    
    # Use log-sum-exp style computation for numerical stability
    log_2pi = np.log(2 * np.pi)
    log_sigma = torch.log(sigma)
    
    # Compute normalized squared residuals
    normalized_sq_residuals = (residuals / sigma) ** 2
    
    # Clamp to prevent extreme values
    normalized_sq_residuals = torch.clamp(normalized_sq_residuals, max=1e3)
    
    log_prob_per_point = -0.5 * (log_2pi + 2 * log_sigma + normalized_sq_residuals)
    
    # DIVING TO GET AVERAGE LOG LIKELIHOOD WHY ? 
    return torch.sum(log_prob_per_point) / x.shape[0]          # average

def log_bernoulli_likelihood(y_expert_edge, soft_gmat_entry, rho, jitter=1e-5):
    """/
    Log likelihood for a single expert edge belief.
    y_expert_edge: scalar tensor, 0 or 1, expert's belief about edge presence.
    soft_gmat_entry: scalar tensor, probability of edge from soft_gmat.
    rho: expert's error rate.
    """
    # p_tilde is the probability of observing y_expert_edge given g_ij (soft_gmat_entry) and rho
    # If y_expert_edge == 1 (expert says edge exists):
    #   P(y=1|g_ij) = g_ij*(1-rho) + (1-g_ij)*rho = rho + g_ij - 2*rho*g_ij (if g_ij is P(edge=1))
    # If y_expert_edge == 0 (expert says edge absent):
    #   P(y=0|g_ij) = g_ij*rho + (1-g_ij)*(1-rho) = 1 - rho - g_ij + 2*rho*g_ij
    # The JAX code uses: p_tilde = rho + g_ij - 2 * rho * g_ij
    # and then loglik = y * log(1-p_tilde) + (1-y) * log(p_tilde)
    # This implies p_tilde is P(expert is wrong OR edge state contradicts expert if expert was right)
    # Let's re-derive based on common interpretation:
    # P(expert_says_1 | true_edge_1) = 1 - rho
    # P(expert_says_1 | true_edge_0) = rho
    # P(expert_says_0 | true_edge_1) = rho
    # P(expert_says_0 | true_edge_0) = 1 - rho
    # log P(expert_y | g_ij) = expert_y * log(P(says_1|g_ij)) + (1-expert_y) * log(P(says_0|g_ij))
    # P(says_1|g_ij) = g_ij * (1-rho) + (1-g_ij) * rho
    # P(says_0|g_ij) = g_ij * rho     + (1-g_ij) * (1-rho)

    prob_expert_says_1 = soft_gmat_entry * (1.0 - rho) + (1.0 - soft_gmat_entry) * rho
    prob_expert_says_0 = soft_gmat_entry * rho + (1.0 - soft_gmat_entry) * (1.0 - rho)

    loglik = y_expert_edge * torch.log(prob_expert_says_1 + jitter) + \
             (1.0 - y_expert_edge) * torch.log(prob_expert_says_0 + jitter)
    return loglik


def scores(z, alpha_hparam):
    u = z[..., 0]  # [..., D, K] 
    v = z[..., 1]  # [..., D, K]
    # Correct: for each (i,j) pair, compute u_i^T v_j
    raw_scores = alpha_hparam * torch.einsum('...ik,...jk->...ij', u, v)
    
    # Mask diagonal properly
    *batch_dims, d, _ = z.shape[:-1]
    diag_mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    if batch_dims:
        diag_mask = diag_mask.expand(*batch_dims, d, d)
    
    return raw_scores * diag_mask



def bernoulli_soft_gmat(z, hparams):
    #"""Return P(edge_ij = 1 | z).  Shape [..., D, D], values in (0,1)."""
    #return torch.sigmoid(scores(z, hparams["alpha"]))
    # Add the mask
    """Return ``P(edge_ij = 1 | z)`` with zero diagonal."""
    probs = torch.sigmoid(scores(z, hparams["alpha"]))
    d = probs.shape[-1]
    diag_mask = 1.0 - torch.eye(d, device=probs.device, dtype=probs.dtype)
    if probs.ndim == 3:
        diag_mask = diag_mask.expand(probs.shape[0], d, d)
    return probs * diag_mask

def bernoulli_sample_gmat(z, hparams):
    """Sample a hard graph using the above probabilities."""
    probs = bernoulli_soft_gmat(z, hparams)
    return torch.bernoulli(probs)


def gumbel_soft_gmat(z, hparams, device=None):
    """
    Works for z shape [..., D, K, 2]  (any number of leading batch dims).
    """
    device  = z.device if device is None else device
    *batch, d, _, _ = z.shape

    # raw scores: [..., D, D]
    raw = scores(z, hparams['alpha'])

    # logistic noise, same shape as raw
    u   = torch.rand(*batch, d, d, device=device, dtype=z.dtype)
    noise = torch.log(u) - torch.log1p(-u)

    soft = torch.sigmoid((noise + raw) * hparams['tau'])

    diag = 1. - torch.eye(d, device=device, dtype=z.dtype)
    if batch:
        diag = diag.expand(*batch, d, d)
    return soft * diag




def gumbel_soft_gmat_v0(z, hparams, device=None):
    if device is None:
        device = z.device
    
    d = z.shape[0]
    raw_scores = scores(z, hparams['alpha'])
    
    # Proper logistic sampling
    u = torch.rand(d, d, device=device, dtype=z.dtype)
    logistic_noise = torch.log(u) - torch.log(1 - u)
    
    # Gumbel-Sigmoid with proper broadcasting
    gumbel_input = (logistic_noise + raw_scores) * hparams['tau']
    soft_gmat = torch.sigmoid(gumbel_input)
    
    # Proper diagonal masking
    diag_mask = 1.0 - torch.eye(d, device=device, dtype=z.dtype)
    return soft_gmat * diag_mask


def log_full_likelihood(data, soft_gmat, current_theta, hparams, device='cpu'):
    """
    Computes log P(Data | Z, Theta).
    data: dict containing 'x' (observational data [N,D]) and optionally 'y' (expert edges list of [i,j,val])
    soft_gmat: The soft adjacency matrix [D,D], typically generated by gumbel_soft_gmat or bernoulli_soft_gmat.
    current_theta: parameters Theta for the current particle [D, D] (for linear model)
    hparams: dict of hyperparameters
    """
    # For likelihood, typically use Bernoulli soft gmat (probabilities)
    # or Gumbel soft gmat if that's part of the likelihood definition (e.g. for certain gradient paths)
    # The JAX log_joint uses bernoulli_soft_gmat for the likelihood term.
    # The JAX grad_z_log_joint_gumbel uses gumbel_soft_gmat for its internal log_density_z.
    # Let's assume for calculating the likelihood value, bernoulli_soft_gmat is appropriate.
    # If this function is used inside a gradient computation that requires Gumbel, that needs to be handled.
    # The `log_density_z` in JAX's `grad_z_log_joint_gumbel` passes `gumbel_soft_gmat(k, z, ...)`
    # to `log_full_likelihood`. So this function needs to accept a pre-computed soft_gmat.

    # Let's modify signature to accept soft_gmat directly,
    # as the caller (`log_density_z` or `log_joint`) will decide which type of soft_gmat.
    # def log_full_likelihood(data, soft_gmat_from_z, current_theta, hparams, device='cpu'):

    # For now, let's stick to the original call and decide inside or assume it's for log_joint value
    # The JAX `log_joint` uses `bernoulli_soft_gmat` for its likelihood.
    # The JAX `grad_z_log_joint_gumbel`'s `log_density_z` uses `gumbel_soft_gmat`.
    # This means `log_full_likelihood` must be flexible or called with the correct `soft_gmat`.
    # Let's assume the `soft_gmat` argument is passed in, matching `log_density_z`'s usage.
    # This function is called by `log_density_z` (for dZ) and `theta_log_joint` (for dTheta).
    # `log_density_z` uses `gumbel_soft_gmat`.
    # `theta_log_joint` uses `bernoulli_soft_gmat`.
    # So, the `soft_gmat` argument is essential.

    # Let's assume the signature is:
    # def log_full_likelihood(data, soft_gmat_arg, current_theta, hparams, device='cpu'):
    # And the original JAX code was:
    # log_full_likelihood(data, soft_gmat=gumbel_soft_gmat(k, z, ...), theta=nonopt_params["theta"], ...)
    # This means the `soft_gmat` argument in the JAX code is the result of `gumbel_soft_gmat` or `bernoulli_soft_gmat`.
    # So, the PyTorch version should also expect `soft_gmat_arg`.
    # The original `dibs.py` has `log_full_likelihood(data, soft_gmat, hard_gmat, theta, hparams)`
    # The `hard_gmat` argument is often None.

    # Let's stick to the JAX function signature as much as possible.
    # `soft_gmat` here is the one passed by the caller.
    # `hard_gmat` is often None.

    # Renaming arguments for clarity based on JAX version
    # def log_full_likelihood(data_dict, soft_gmat_tensor, hard_gmat_tensor_or_none, theta_tensor, hparams_dict, device='cpu'):
    
    # Using names from the original JAX function for consistency in this file
    # data, soft_gmat, hard_gmat (often None), theta, hparams

    # Observational likelihood (Gaussian)
    # pred_mean = data['x'] @ (current_theta * soft_gmat) # Assumes data['x'] is [N, D]
    # The JAX code uses data["x"] @ (theta * soft_gmat)
    # This implies data['x'] is [N, D], theta is [D, D], soft_gmat is [D, D]
    # Resulting pred_mean is [N, D]
    # This is for a linear model where theta contains coefficients.
    # For PyTorch, ensure data['x'] is a tensor.
    x_data = data['x'] # Shape [N, D]
    
    # Effective weighted adjacency matrix
    # Theta contains potential coefficients, soft_gmat gates/weights them
    effective_W = current_theta * soft_gmat # [D, D]
    
    # Predicted mean: X_pred = X_obs @ W_eff
    # This is a common formulation for Structural Equation Models / linear autoregressive processes
    # X_pred_i = sum_j X_obs_j * W_eff_ji
    pred_mean = torch.matmul(x_data, effective_W) # [N, D]

    # sigma_obs is a hyperparameter, e.g., hparams.get('sigma_obs_noise', 0.1)
    sigma_obs = hparams.get('sigma_obs_noise', 0.1) # Get from hparams or default
    log_obs_likelihood = log_gaussian_likelihood(x_data, pred_mean, sigma=sigma_obs)

    log_expert_likelihood_total = torch.tensor(0.0, device=device)
    inv_temperature_expert = 0.0

    if data.get("y", None) is not None and len(data["y"]) > 0:
        inv_temperature_expert = hparams.get("temp_ratio", 0.0) # JAX uses hparams["temp_ratio"]
        
        # data["y"] is a list of [i, j, val]
        # We need to iterate through these expert beliefs.
        expert_log_probs = []
        for expert_belief in data["y"]:
            i, j, val = int(expert_belief[0]), int(expert_belief[1]), expert_belief[2].item() # Ensure indices are int
            
            # Get the corresponding P(edge_ij=1) from the soft_gmat
            g_ij_prob = soft_gmat[i, j]
            
            # Calculate log likelihood for this single expert belief
            # Ensure val is a tensor for log_bernoulli_likelihood
            y_val_tensor = torch.tensor(val, dtype=g_ij_prob.dtype, device=g_ij_prob.device)
            expert_log_probs.append(
                log_bernoulli_likelihood(y_val_tensor, g_ij_prob, hparams["rho"])
            )
        
        if expert_log_probs:
            log_expert_likelihood_total = torch.sum(torch.stack(expert_log_probs))

    return inv_temperature_expert * log_expert_likelihood_total + log_obs_likelihood


def log_theta_prior(theta_effective, theta_prior_mean, theta_prior_sigma):
    """
    Log prior probability of Theta (Gaussian).
    theta_effective: The parameters being penalized (e.g., theta * G_soft)
    theta_prior_mean: Mean of the Gaussian prior (often zeros)
    theta_prior_sigma: Std dev of the Gaussian prior
    """
    # Ensure theta_prior_mean is a tensor with compatible shape or broadcastable
    if not torch.is_tensor(theta_prior_mean):
        theta_prior_mean = torch.full_like(theta_effective, theta_prior_mean)
        
    return log_gaussian_likelihood(theta_effective, theta_prior_mean, sigma=theta_prior_sigma)




def gumbel_acyclic_constr_mc(z_particle, d, hparams, n_mc_samples, device='cpu'):
    """
    IT Should take hard graphs for h 
        Or, if h(G) is defined on hard graphs, E_{G_hard ~ Bernoulli(G_soft)} [h(G_hard)].
    z_particle: single Z particle [D, K, 2]
    d: number of nodes
    hparams: hyperparameters
    n_mc_samples: number of MC samples for the expectation
    """
    h_samples = []
    for _ in range(n_mc_samples):
        g_soft = gumbel_soft_gmat(z_particle, hparams, device=device)
        
        #  we should give hard gmat to get h values
        h_samples.append(acyclic_constr(torch.bernoulli(g_soft), d))
        # h_samples.append(acyclic_constr(g_soft, d))
    

    return torch.mean(torch.stack(h_samples))


# --- Gradient Computations ---
# These are the most complex to translate due to JAX's `grad` and `vmap`.
# We'll use `torch.autograd.grad` and manual looping or batching for vmap.

def gumbel_grad_acyclic_constr_mc(z_particle_opt, d, hparams_dict_nonopt, n_mc_samples, create_graph=False):
    """
    Computes E_L [grad_Z h(G_soft(Z,L))]
    z_particle_opt: The Z tensor [D,K,2] for which we need gradients. Must have requires_grad=True.
    d: number of nodes
    hparams_dict_nonopt: hyperparameters
    n_mc_samples: number of MC samples for the expectation
    create_graph: bool, for torch.autograd.grad
    """
    # This function computes the gradient of E_L[h(G_soft(Z,L))] w.r.t Z.
    # = E_L[grad_Z h(G_soft(Z,L))]
    # We need to sample L, compute G_soft, compute h(G_soft), then get grad_Z h(Gsoft).
    
    grad_h_samples = []
    for _ in range(n_mc_samples):
        # Ensure z_particle_opt is a leaf and requires grad for this specific sample's computation
        # If z_particle_opt is already a parameter being optimized by an outer loop, this is fine.
        # If it's an intermediate tensor, it might need to be cloned and set to require_grad.
        # For now, assume z_particle_opt is the parameter we are differentiating.

        # Generate G_soft(Z,L). This operation must be part of the graph for autograd.
        g_soft = gumbel_soft_gmat(z_particle_opt, hparams_dict_nonopt, device=z_particle_opt.device)
        h_val = acyclic_constr(g_soft, d)
        
        # Compute gradient of h_val w.r.t. z_particle_opt for this one MC sample
        # `h_val` is a scalar. `z_particle_opt` is the tensor of interest.
        grad_h_val_wrt_z, = torch.autograd.grad(
            outputs=h_val,
            inputs=z_particle_opt,
            grad_outputs=torch.ones_like(h_val), # for scalar output
            retain_graph=True, # Potentially needed if z_particle_opt is used in multiple h_samples
                               # or if the graph is needed later. Set to False if not.
                               # If this is the only use of z_particle_opt for this h_val, False is fine.
                               # For an expectation, we sum gradients, so True is safer if z_particle_opt is the same object.
            create_graph=create_graph # If we need higher-order derivatives
        )
        grad_h_samples.append(grad_h_val_wrt_z)

    if not grad_h_samples:
        return torch.zeros_like(z_particle_opt)
        
    # Average the gradients
    # The JAX code uses jnp.mean(..., axis=0)
    # Here, stack and mean over the sample dimension (dim=0)
    avg_grad_h = torch.mean(torch.stack(grad_h_samples), dim=0)
    return avg_grad_h


def grad_z_log_joint_gumbel(current_z_opt, current_theta_nonopt, data_dict, hparams_full, device='cpu'):
    """
    Computes gradient of log P(Z, Theta, Data) w.r.t. Z. (Eq 10 in DiBS paper, adapted)
    current_z_opt: Z tensor [D,K,2] requiring grad.
    current_theta_nonopt: Theta tensor [D,D], treated as fixed for this grad computation.
    data_dict: Data.
    hparams_full: Combined hyperparameters and current non-optimal params (like current_theta_nonopt).
    """
    d = current_z_opt.shape[0]
    beta = hparams_full['beta']
    sigma_z_sq = hparams_full['sigma_z']**2
    n_grad_mc_samples = hparams_full['n_grad_mc_samples'] # For likelihood part

    # 1. Gradient of log prior on Z: grad_Z [ -beta * E[h(G_soft)] - 0.5/sigma_z^2 * ||Z||^2 ]
    # 1a. grad_Z E[h(G_soft)]
    # The JAX code uses `gumbel_grad_acyclic_constr_mc` which averages grads.
    # n_nongrad_mc_samples from hparams is used by JAX's gumbel_grad_acyclic_constr_mc.
    n_acyclic_mc_samples = hparams_full.get('n_nongrad_mc_samples', hparams_full['n_grad_mc_samples'])
    grad_expected_h = gumbel_grad_acyclic_constr_mc(
        current_z_opt, d, hparams_full, n_acyclic_mc_samples, create_graph=False # No higher order grads needed here
    )
    grad_log_z_prior_acyclicity = -beta * grad_expected_h
    
    # 1b. grad_Z [- 1/(2*sigma_z^2) * sum(Z_flat^2) ] = - Z / sigma_z^2
    # (Assuming prior is N(0, sigma_z^2) for each element of Z, so logpdf is -0.5*z^2/sigma_z^2 - log(sigma_z*sqrt(2pi)))
    # The JAX code has `-(1 / nonopt_params["sigma_z"] ** 2) * opt_params["z"]`. This implies sum over elements.
    # If prior is sum_elements N(0, sigma_z), then log prior is sum (-0.5 * (z_ij/sigma_z)^2)
    # grad is -z_ij / sigma_z^2.
    grad_log_z_prior_gaussian = -current_z_opt / sigma_z_sq
    
    grad_log_z_prior_total = grad_log_z_prior_acyclicity + grad_log_z_prior_gaussian

    log_p_samples_list = []

    for i in range(n_grad_mc_samples):
        # Ensure z_opt is used in a way that its gradient can be taken for this sample
        # If current_z_opt is a parameter, it's fine.
        
        # G_soft for this MC sample, depends on current_z_opt and fresh Logistic noise
        # This G_soft must be constructed in a way that tracks gradients back to current_z_opt
        g_soft_mc = gumbel_soft_gmat(current_z_opt, hparams_full, device=device) # differentiable wrt current_z_opt
        
        # Calculate log P(Data | G_soft, Theta_nonopt)
        log_lik_val = log_full_likelihood(data_dict, g_soft_mc, current_theta_nonopt, hparams_full, device=device)
        
        # Calculate log P(Theta_eff | G_soft)
        # Theta_eff = Theta_nonopt * G_soft
        theta_eff_mc = current_theta_nonopt * g_soft_mc
        # Prior mean and sigma for theta from hparams
        theta_prior_mean_val = torch.zeros_like(current_theta_nonopt, device=device) # Example
        theta_prior_sigma_val = hparams_full.get('theta_prior_sigma', 1.0)
        log_theta_prior_val = log_theta_prior(theta_eff_mc, theta_prior_mean_val, theta_prior_sigma_val)
        
        log.debug(f"---- log_theta_prior_val ---- Shape: {log_theta_prior_val.shape}, Value: {log_theta_prior_val}")
        log.debug(f"---- log_lik_val ---- Shape: {log_lik_val.shape}, Value: {log_lik_val}")

        current_log_density = log_lik_val + log_theta_prior_val
        log_p_samples_list.append(current_log_density) # Detach for storing value, grad comes next



    # Stack the collected tensors
    log_p_samples = torch.stack(log_p_samples_list, dim=0) # [n_grad_mc_samples]
    mean_log_p_samples = log_p_samples.mean()
    grad_curr_log_density_wrt_z, = torch.autograd.grad(outputs=mean_log_p_samples, inputs=current_z_opt)


    # To avoid numerical issues with exp(log_p_samples):
    log_p_max = torch.max(log_p_samples)
    shifted_log_p = log_p_samples - log_p_max # For stability
    exp_shifted_log_p = torch.exp(shifted_log_p) # These are proportional to actual probabilities

    # Numerator term for each sample: exp_shifted_log_p[s] * grad_log_p_wrt_z_samples[s]
    # Sum these up, then divide by sum(exp_shifted_log_p)
    
    # Reshape for broadcasting: exp_shifted_log_p needs to be [N, 1, 1, 1]
    exp_shifted_log_p_reshaped = exp_shifted_log_p.reshape(-1, *([1]*(current_z_opt.ndim)))

    numerator_sum = torch.sum(exp_shifted_log_p_reshaped * grad_curr_log_density_wrt_z, dim=0)
    denominator_sum = torch.sum(exp_shifted_log_p)

    if denominator_sum < 1e-33: # Avoid division by zero
        grad_log_likelihood_part = torch.zeros_like(current_z_opt)
    else:
        grad_log_likelihood_part = numerator_sum / denominator_sum
        
    return grad_log_z_prior_total + grad_log_likelihood_part


def grad_theta_log_joint(current_z_nonopt, current_theta_opt, data_dict, hparams_full, device='cpu'):
    """
    Computes gradient of log P(Z, Theta, Data) w.r.t. Theta. (Eq 11 in DiBS paper, adapted)
    current_z_nonopt: Z tensor [D,K,2], treated as fixed.
    current_theta_opt: Theta tensor [D,D] requiring grad.
    data_dict: Data.
    hparams_full: Combined hyperparameters.
    """
    # d = current_z_nonopt.shape[0] # This variable was unused
    n_grad_mc_samples = hparams_full['n_grad_mc_samples']

    # For grad_theta, the paper (Eq A.33) suggests using Bernoulli soft gmat (not Gumbel).
    # This is because Theta's gradient doesn't involve differentiating the graph sampling process itself,
    # but rather differentiating the likelihood and theta prior for a given graph distribution from Z.
    # The expectation is over G ~ Bernoulli(sigma(Z)).
    # The JAX code's `theta_log_joint` lambda uses `bernoulli_soft_gmat(nonopt_params["z"], ...)`


    log_density_samples_list = []

    # The `bernoulli_soft_gmat` depends on Z, which is fixed here.
    # The original code recomputed g_soft_from_fixed_z in the loop. This behavior is maintained.
    # This loop collects log P(Theta, Data | G_sample) for M samples.
    # G_sample might be deterministic (if bernoulli_soft_gmat is used directly and is deterministic)
    # or stochastic (if G is sampled from bernoulli_soft_gmat, or if bernoulli_soft_gmat itself is stochastic).
    # The current interpretation, based on JAX code comments, is that bernoulli_soft_gmat is used directly.
    for i in range(n_grad_mc_samples):
        # TODO look at the expectectation for mc (Original comment, relevant to g_soft generation strategy)
        g_soft_from_fixed_z = bernoulli_soft_gmat(current_z_nonopt, hparams_full) 
        
        
        # TRY WITH sampling g   
        #g_soft_for_lik = g_soft_from_fixed_z 
        g_soft_for_lik = torch.bernoulli(g_soft_from_fixed_z)
        ## IMPORTANT TODO THIS IS NOT SOFT CHANGE THE NAME 

        log_lik_val = log_full_likelihood(data_dict, g_soft_for_lik, current_theta_opt, hparams_full, device=device)
        
        theta_eff_mc = current_theta_opt * g_soft_for_lik # Effective theta based on this g_soft
        theta_prior_mean_val = torch.zeros_like(current_theta_opt, device=device)
        theta_prior_sigma_val = hparams_full.get('theta_prior_sigma', 1.0)
        log_theta_prior_val = log_theta_prior(theta_eff_mc, theta_prior_mean_val, theta_prior_sigma_val)
        
        current_log_density = log_lik_val + log_theta_prior_val
        # Append the tensor itself, maintaining the computation graph. Do NOT detach.
        log_density_samples_list.append(current_log_density)

    # Stack all collected log_density samples
    stacked_log_densities = torch.stack(log_density_samples_list)

    log.debug(f"---- stacked_log_densities ---- Shape: {stacked_log_densities.shape}")
    log.debug(f"---- stacked_log_densities ---- Values: {stacked_log_densities}")
        
    avg_log_density = torch.mean(stacked_log_densities)
    
    log.debug(f"---- avg_log_density ---- Shape: {avg_log_density.shape}, Value: {avg_log_density}")
    # Log current_theta_opt and log_lik_val before gradient calculation
    log.debug(f"---- current_theta_opt (before grad) ---- Shape: {current_theta_opt.shape}, Values:\n{current_theta_opt.detach().round(decimals=4)}")
    log.debug(f"---- log_lik_val (sample from mean calculation) ---- Shape: {log_lik_val.shape}, Value: {log_lik_val.detach().round(decimals=4)}")

            
    grad_avg_log_density_wrt_theta, = torch.autograd.grad(
        outputs=avg_log_density,
        inputs=current_theta_opt,
    )
    
    log.debug(f"---- grad_avg_log_density_wrt_theta ---- Shape: {grad_avg_log_density_wrt_theta.shape}, Value: {grad_avg_log_density_wrt_theta}")
        
    return grad_avg_log_density_wrt_theta


def grad_log_joint(params, data_dict, hparams_dict_config, device='cpu', **annealing_kwargs):
    """
    Computes gradients of the log joint P(Z, Theta, Data) w.r.t Z and Theta.
    **annealing_kwargs: Additional parameters for update_dibs_hparams
    """
    # Ensure params require grad
    current_z = params['z']
    current_theta = params['theta']
    
    if not current_z.requires_grad:
        current_z.requires_grad_(True)
    if not current_theta.requires_grad:
        current_theta.requires_grad_(True)

    t_anneal = params.get('t', torch.tensor([0.0], device=device)).item()
    
    # Update hparams with annealing - pass through kwargs
    hparams = update_dibs_hparams(hparams_dict_config, t_anneal, **annealing_kwargs)

    # Compute grad_z
    grad_z = grad_z_log_joint_gumbel(
        current_z_opt=current_z,
        current_theta_nonopt=current_theta.detach(),
        data_dict=data_dict,
        hparams_full=hparams,
        device=device
    )
    
    # Compute grad_theta
    grad_theta = grad_theta_log_joint(
        current_z_nonopt=current_z.detach(),
        current_theta_opt=current_theta,
        data_dict=data_dict,
        hparams_full=hparams,
        device=device
    )
    
    return {"z": grad_z, "theta": grad_theta, "t": torch.tensor([0.0], device=device)}


def log_joint(params, data_dict, hparams_dict_config, device='cpu', **annealing_kwargs):
    """
    Computes the log joint density log P(Z, Theta, Data).
    **annealing_kwargs: Additional parameters for update_dibs_hparams
    """
    current_z = params['z']
    current_theta = params['theta']
    t_anneal = params.get('t', torch.tensor([0.0], device=device)).item()

    # Update hparams with annealing - pass through kwargs
    hparams_updated = update_dibs_hparams(hparams_dict_config.copy(), t_anneal, **annealing_kwargs)
    
    d = current_z.shape[0] # Assuming z is [D, K, 2]

    # 1. Log Likelihood part: log P(Data | Z, Theta)
    #    Uses bernoulli_soft_gmat as per JAX log_joint
    g_soft_for_lik = bernoulli_soft_gmat(current_z, hparams_updated)
    log_lik = log_full_likelihood(data_dict, g_soft_for_lik, current_theta, hparams_updated, device=device)

    # 2. Log Prior part: log P(Z) + log P(Theta | Z)
    # 2a. Log P(Z) = log P_gaussian(Z) + log P_acyclic(Z)
    #     log P_gaussian(Z)
    log_prior_z_gaussian = torch.sum(Normal(0.0, hparams_updated['sigma_z']).log_prob(current_z))
    
    #     log P_acyclic(Z) = -beta * E_G_soft~p(G|Z) [h(G_soft)]
    #     The JAX log_joint uses `gumbel_acyclic_constr_mc` for the prior.
    #     This uses Gumbel soft matrices for the expectation of h(G).
    n_acyclic_mc_samples = hparams_updated.get('n_nongrad_mc_samples', hparams_updated['n_grad_mc_samples'])
    expected_h_val = gumbel_acyclic_constr_mc(
        current_z, d, hparams_updated, n_acyclic_mc_samples, device=device
    )
    log_prior_z_acyclic = -hparams_updated['beta'] * expected_h_val
    
    log_prior_z_total = log_prior_z_gaussian + log_prior_z_acyclic

    # 2b. Log P(Theta | Z)
    #     Prior on theta_effective = theta * G_soft_bernoulli
    g_soft_for_theta_prior = bernoulli_soft_gmat(current_z, hparams_updated)
    theta_eff = current_theta * g_soft_for_theta_prior
    theta_prior_mean = torch.zeros_like(current_theta, device=device) # Example
    theta_prior_sigma = hparams_updated.get('theta_prior_sigma', 1.0)
    log_prior_theta_given_z = log_theta_prior(theta_eff, theta_prior_mean, theta_prior_sigma)
    
    total_log_prior = log_prior_z_total + log_prior_theta_given_z
    
    return log_lik + total_log_prior



def update_dibs_hparams(
    hparams_dict: dict,
    t_step: int | float,
    *,
    # ---- schedule knobs you can tune from Hydra ---------------------------
    alpha_final: float = 5.0,      # target α after the warm-up
    alpha_warmup: int  = 500,      # iterations over which α grows
    beta_final:  float = 50.0,     # target β
    beta_delay:  int   = 250,      # keep β frozen for the first β_delay iters
    beta_ramp:   int   = 750,      # length of the β linear ramp
    tau_final:   float | None = None,  # optional τ cooling
) -> dict:
    """
    Anneal α *early* (soft-edges → hard) and β *later* (loose → strict DAG),
    as recommended in DiBS §3.3 / Alg. 1.

    • α grows **linearly** from its initial value to `alpha_final`
      over `alpha_warmup` iterations.

    • β stays constant until `beta_delay`, then grows linearly to
      `beta_final` over `beta_ramp` iterations.

    • τ can be cooled in the same fashion if `tau_final` is given.
    """
    hp = hparams_dict.copy()          # do **not** mutate the original dict
    t  = float(t_step)

    # -------- α schedule ---------------------------------------------------
    #   α_t = α_0  +  (α* − α_0) · clip(t / warmup, 0, 1)
    alpha0 = hparams_dict["alpha"]
    alpha_progress = min(max(t / alpha_warmup, 0.0), 1.0)
    hp["alpha"] = alpha0 + (alpha_final - alpha0) * alpha_progress

    # -------- β schedule ---------------------------------------------------
    #   β_t = β_0  (t < delay)
    #       = β_0  +  (β* − β_0) · clip((t-delay) / ramp, 0, 1)
    beta0 = hparams_dict["beta"]
    if t < beta_delay:
        hp["beta"] = beta0
    else:
        beta_progress = min(max((t - beta_delay) / beta_ramp, 0.0), 1.0)
        hp["beta"] = beta0 + (beta_final - beta0) * beta_progress

    # -------- optional τ cooling ------------------------------------------
    if tau_final is not None:
        tau0 = hparams_dict["tau"]
        # cool τ on the *same* timeline as α so gradients do not vanish too soon
        hp["tau"] = tau0 + (tau_final - tau0) * alpha_progress

    return hp





def hard_gmat_particles_from_z(z_particles, alpha_hparam_for_scores=1.0):
    """
    Converts Z particles to hard adjacency matrices by thresholding scores.
    z_particles: [N_particles, D, K, 2]
    alpha_hparam_for_scores: alpha used in scores() function.
    Returns: hard_gmat_particles [N_particles, D, D]
    """
    # scores() computes alpha * u^T v and masks diagonal
    # For hard_gmat, we typically use the raw scores before sigmoid.
    # The JAX `hard_gmat_particles_from_z` uses `scores(z, d, {"alpha":1.0}) > 0.0`.
    # This implies alpha=1.0 for this specific conversion, or it's passed.
    
    # Get scores for each particle
    # `scores` function handles batching if z_particles is [N, D, K, 2]
    s = scores(z_particles, alpha_hparam=alpha_hparam_for_scores) # [N, D, D]
    
    hard_gmats = (s > 0.0).to(torch.float32) # Convert boolean to float (0.0 or 1.0)
    return hard_gmats


class Logistic(Distribution):
    """
    Logistic distribution implementation for PyTorch.
    """
    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, torch.Tensor):
            batch_shape = self.loc.size()
        else:
            batch_shape = torch.Size()
        super(Logistic, self).__init__(batch_shape, validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * (torch.log(u) - torch.log(1 - u))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) / self.scale
        return -z - 2 * torch.log(1 + torch.exp(-z)) - torch.log(self.scale)