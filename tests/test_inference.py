import torch
import torch.nn.functional as F
from torch.distributions import Normal
import math
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Get the directory of the current script
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (assuming tests are in a subdir of project root)
project_root = os.path.join(current_script_dir, '..')
# Add the project root to sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.dibs_torch_v2 import (
    log_joint, grad_log_joint, bernoulli_soft_gmat,
    hard_gmat_particles_from_z, log_gaussian_likelihood,
    log_bernoulli_likelihood, log_full_likelihood, log_theta_prior,
    gumbel_soft_gmat, gumbel_grad_acyclic_constr_mc,
    grad_z_log_joint_gumbel, grad_theta_log_joint,
    update_dibs_hparams, scores, acyclic_constr
)
# Placeholder for utils if needed, e.g., create_dummy_z, create_dummy_theta
# from models.utils_torch import sample_x # If used

# Get a logger for this module
log = logging.getLogger("InferTest")

# --- Utility Functions for Testing (can be adapted from test_dibs_torch.py) ---
def create_dummy_z(d=3, k=2, requires_grad=False, seed=None):
    """Creates a dummy Z tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(d, k, 2)
    if requires_grad:
        z.requires_grad_(True)
    return z

def create_dummy_theta(d=3, requires_grad=False, seed=None):
    """Creates a dummy Theta tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    theta = torch.randn(d, d) # Corrected from (d, k, 2) to (d,d)
    if requires_grad:
        theta.requires_grad_(True)
    return theta

def create_dummy_hparams(d=3, alpha_val=1.0, beta_val=1.0, tau_val=1.0, sigma_z_val=1.0,
                         sigma_obs_noise_val=0.1, rho_val=0.05, temp_ratio_val=1.0,
                         n_grad_mc_samples_val=2, n_nongrad_mc_samples_val=2,
                         theta_prior_sigma_val=1.0):
    """Creates a dummy hparams dictionary."""
    return {
        'alpha': alpha_val,
        'beta': beta_val,
        'tau': tau_val,
        'sigma_z': sigma_z_val,
        'sigma_obs_noise': sigma_obs_noise_val,
        'rho': rho_val,
        'temp_ratio': temp_ratio_val,
        'n_grad_mc_samples': n_grad_mc_samples_val,
        'n_nongrad_mc_samples': n_nongrad_mc_samples_val,
        'theta_prior_sigma': theta_prior_sigma_val,
        'd': d,
    }


def generate_ground_truth_data_x1_x2_x3(num_samples, obs_noise_std, seed=None):
    """
    Generates data for the ground truth causal chain X1 -> X2 -> X3.
    X1 ~ N(0, 1) (Exogenous)
    X2 = 2.0 * X1 + N(0, obs_noise_std^2)
    X3 = -1.5 * X2 + N(0, obs_noise_std^2)
    """
    if seed is not None:
        torch.manual_seed(seed)

    D_nodes = 3
    
    # Define Ground Truth Graph (Adjacency Matrix)
    # G_true[i, j] = 1 if i -> j
    G_true = torch.zeros(D_nodes, D_nodes, dtype=torch.float32)
    G_true[0, 1] = 1.0  # X1 -> X2
    G_true[1, 2] = 1.0  # X2 -> X3

    # Define Ground Truth Coefficients (Theta)
    # Theta_true[i, j] is the coefficient for X_i in the equation for X_j
    Theta_true = torch.zeros(D_nodes, D_nodes, dtype=torch.float32)
    Theta_true[0, 1] = 2.0   # X2 = 2.0 * X1 + noise
    Theta_true[1, 2] = -1.5  # X3 = -1.5 * X2 + noise

    # Generate Data
    X_data = torch.zeros(num_samples, D_nodes)

    # X1 (node 0) is exogenous, sample from N(0,1) or N(0, obs_noise_std^2)
    # For simplicity and to ensure it has some variance not solely dependent on obs_noise_std if obs_noise_std is very small for structural equations.
    # Let's assume X1's intrinsic variance is 1.0, and obs_noise_std applies to the structural equations.
    X_data[:, 0] = torch.randn(num_samples) # * 1.0 (stddev of 1)

    # X2 (node 1) = Theta_true[0,1] * X1 + noise
    noise_x2 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 1] = Theta_true[0, 1] * X_data[:, 0] + noise_x2

    # X3 (node 2) = Theta_true[1,2] * X2 + noise
    noise_x3 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 2] = Theta_true[1, 2] * X_data[:, 1] + noise_x3
    
    return {'x': X_data, 'G_true': G_true, 'Theta_true': Theta_true, 'y': None} # 'y' for expert data, None here

# test_inference.py
import torch
# ... other imports from your script ...
from models.dibs_torch_v2 import ( # Ensure all necessary model functions are imported
    log_joint, grad_log_joint, bernoulli_soft_gmat,
    hard_gmat_particles_from_z, update_dibs_hparams #
)
# ... utility functions like create_dummy_z, create_dummy_theta, generate_ground_truth_data_x1_x2_x3 ...
# Make sure these utility functions are defined or imported correctly.

import hydra
from omegaconf import DictConfig, OmegaConf
import math # For math.sqrt if used in sigma_z calculation

# --- Utility Functions for Testing (Copied/Adapted from your script) ---
# Ensure create_dummy_z, create_dummy_theta, generate_ground_truth_data_x1_x2_x3
# and create_dummy_hparams (or an adapted version) are defined here or imported.

# Adapted create_dummy_hparams to take cfg and d_nodes, k_latent
def create_model_hparams_from_cfg(cfg: DictConfig, d_nodes: int, k_latent: int) -> dict:
    sigma_z_val = (1.0 / math.sqrt(k_latent)) / cfg.model_hparams.sigma_z_val_divisor_for_k_latent

    sigma_obs_noise = cfg.data.synthetic_obs_noise_std if cfg.model_hparams.sigma_obs_noise_val_is_data_noise else cfg.model_hparams.get("sigma_obs_noise_explicit_val", 0.1)


    return {
        'alpha': cfg.model_hparams.alpha_val,
        'beta': cfg.model_hparams.beta_val,
        'tau': cfg.model_hparams.tau_val,
        'sigma_z': sigma_z_val,
        'sigma_obs_noise': sigma_obs_noise,
        'rho': cfg.model_hparams.rho_val,
        'temp_ratio': cfg.model_hparams.temp_ratio_val,
        'n_grad_mc_samples': cfg.model_hparams.n_grad_mc_samples_val,
        'n_nongrad_mc_samples': cfg.model_hparams.n_nongrad_mc_samples_val,
        'theta_prior_sigma': cfg.model_hparams.theta_prior_sigma_val,
        'd': d_nodes, # Important to pass the actual d_nodes
    }


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_gradient_ascent_experiment(cfg: DictConfig) -> None:
    # Hydra automatically sets up basic logging.
    # Your log messages will go to the console and to a file in the output directory.

    log.info("\n--- Running Gradient Ascent Test with X1->X2->X3 Ground Truth (Hydra Config) ---")
    log.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Define Constants and Generate Ground Truth Data
    # These now come from cfg
    D_nodes = cfg.data.d_nodes
    K_latent = cfg.particle.k_latent
    N_samples_data = cfg.data.num_samples
    synthetic_obs_noise_std = cfg.data.synthetic_obs_noise_std
    ground_truth_seed = cfg.data.ground_truth_seed # Use this for data generation consistency

    # Use a global seed if desired, or specific seeds from config
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    data_package = generate_ground_truth_data_x1_x2_x3( # Assuming this function is defined above
        num_samples=N_samples_data,
        obs_noise_std=synthetic_obs_noise_std,
        seed=ground_truth_seed
    )
    data_dict = {'x': data_package['x'], 'y': data_package['y']}
    G_true = data_package['G_true']
    Theta_true = data_package['Theta_true']

    log.info("Ground Truth Model:")
    log.info(f"G_true:\n{G_true.int()}")
    log.info(f"Theta_true:\n{Theta_true}")
    # ... (rest of your print statements)

    # 2. Initialize Particle (Z and Theta)
    # Allow seeds for Z and Theta to be offset from a main seed or set directly
    z_init_seed = cfg.seed + cfg.particle.init_seed_offset if cfg.seed is not None else None
    theta_init_seed = cfg.seed + cfg.particle.init_seed_offset + 1 if cfg.seed is not None else None

    Z_current = create_dummy_z(D_nodes, K_latent, requires_grad=False, seed=z_init_seed) #
    Theta_current = create_dummy_theta(D_nodes, requires_grad=False, seed=theta_init_seed) #


    # 3. Base Hyperparameters for DiBS
    # create_model_hparams_from_cfg will use cfg and other dynamic values like D_nodes, K_latent
    base_hparams_config = create_model_hparams_from_cfg(cfg, D_nodes, K_latent)


    # 4. Learning Rates and Iterations from cfg
    lr_z = cfg.training.lr_z
    lr_theta = cfg.training.lr_theta
    num_iterations = cfg.training.num_iterations
    max_grad_norm_z = cfg.training.max_grad_norm_z
    max_grad_norm_theta = cfg.training.max_grad_norm_theta
    device = cfg.device

    log.info(f"\nInitial Z (norm): {Z_current.norm().item():.4f}")
    log.info(f"Initial Theta (norm): {Theta_current.norm().item():.4f}")
    log.info(f"Learning rates: lr_z={lr_z}, lr_theta={lr_theta}")
    log.info(f"Iterations: {num_iterations}")
    log.info(f"Device: {device}")
    # ... (rest of your initial print statements)

    # 5. Gradient Ascent Loop (largely the same, uses variables derived from cfg)
    for t_iter in range(num_iterations):
        current_annealing_t_for_hparams = torch.tensor(float(t_iter + 1))
        Z_param = Z_current.clone().detach().requires_grad_(True)
        Theta_param = Theta_current.clone().detach().requires_grad_(True)
        params_for_calc = {'z': Z_param, 'theta': Theta_param, 't': current_annealing_t_for_hparams}

        lj_val = float('nan')
        try:
            lj = log_joint(params_for_calc, data_dict, base_hparams_config, device=device) #
            lj_val = lj.item()
        except Exception as e:
            log.error(f"Error computing log_joint at iter {t_iter+1}: {e}")

        grad_z_norm = float('nan'); grad_theta_norm = float('nan')
        try:
            grads = grad_log_joint(params_for_calc, data_dict, base_hparams_config, device=device) #
            grad_z = grads['z']
            grad_theta = grads['theta']

            if max_grad_norm_z > 0: # Check if clipping is enabled
                torch.nn.utils.clip_grad_norm_(grad_z, max_grad_norm_z)
            if max_grad_norm_theta > 0: # Check if clipping is enabled
                torch.nn.utils.clip_grad_norm_(grad_theta, max_grad_norm_theta)
            
            grad_z_norm = grad_z.norm().item()
            grad_theta_norm = grad_theta.norm().item()
        except Exception as e:
            log.error(f"Error computing gradients at iter {t_iter+1}: {e}")
            break

        with torch.no_grad():
            Z_current.add_(lr_z * grad_z)
            Theta_current.add_(lr_theta * grad_theta)

        if (t_iter + 1) % 10 == 0 or t_iter == 0 or (t_iter + 1) == num_iterations :
            log.info(f"Iter {t_iter+1:3d}: Z_norm={Z_current.norm().item():.4f}, Theta_norm={Theta_current.norm().item():.4f}, "
                  f"log_joint={lj_val:.4f}, grad_Z_norm={grad_z_norm:.4e}, grad_Theta_norm={grad_theta_norm:.4e}")

            current_hparams_for_iter = update_dibs_hparams(base_hparams_config, current_annealing_t_for_hparams.item()) #
            log.info(f"    Annealed: alpha={current_hparams_for_iter['alpha']:.3f}, beta={current_hparams_for_iter['beta']:.3f}, tau={current_hparams_for_iter['tau']:.3f}")
            
            hparams_for_edge_probs = {'alpha': current_hparams_for_iter['alpha'], 'd': D_nodes}
            with torch.no_grad():
                current_edge_probs = bernoulli_soft_gmat(Z_current, hparams_for_edge_probs) #
            log.info(f"    Current Edge Probs (from Z, alpha={hparams_for_edge_probs['alpha']:.3f}):\n{current_edge_probs.round(decimals=2)}")

            if torch.isnan(Z_current).any() or torch.isnan(Theta_current).any() or \
               torch.isinf(Z_current).any() or torch.isinf(Theta_current).any():
                log.error("!!! NaN/Inf detected in parameters. Stopping. !!!")
                break
    
    # ... (rest of your script: final prints, comparisons, assertions)
    # Make sure to use G_true, Theta_true, Z_current, Theta_current, D_nodes, base_hparams_config, num_iterations
    # and functions like hard_gmat_particles_from_z and update_dibs_hparams correctly.
    log.info("\n--- Single Particle Gradient Ascent (X1->X2->X3 Data with Hydra) Completed ---")
    log.info(f"Final Z (norm): {Z_current.norm().item():.4f}")
    log.info(f"Final Theta (norm): {Theta_current.norm().item():.4f}")

    # Comparison with Ground Truth
    final_t_for_hparams = torch.tensor(float(num_iterations))
    final_hparams = update_dibs_hparams(base_hparams_config, final_t_for_hparams.item()) #
    
    G_learned_hard = torch.zeros_like(G_true)
    try:
        alpha_for_hard_gmat = final_hparams.get('alpha', 1.0) 
        G_learned_hard = hard_gmat_particles_from_z( #
            Z_current.unsqueeze(0), 
            alpha_hparam_for_scores=alpha_for_hard_gmat
        ).squeeze(0).int()
    except Exception as e:
        log.error(f"Could not generate final hard graph from Z_current: {e}. Using zeros for G_learned_hard.")

    log.info("\n--- Comparison with Ground Truth ---")
    # ... (your comparison print statements from test_inference.py) ...


if __name__ == '__main__':
    # Remove the direct call to run_gradient_ascent_test_with_x1_x2_x3_data()
    # Hydra will call the function decorated with @hydra.main()
    run_gradient_ascent_experiment()