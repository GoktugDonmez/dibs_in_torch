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

from models.dibs_torch_non_linear import (
    log_joint_mlp, grad_log_joint_mlp, log_full_likelihood_mlp, log_phi_prior,
    NodeFFN, create_dummy_phi_nets, hard_gmat_particles_from_z_mlp
)
from models.dibs_torch_v2 import (
    bernoulli_soft_gmat, hard_gmat_particles_from_z,
    update_dibs_hparams, scores, acyclic_constr
)

# Get a logger for this module
log = logging.getLogger("InferTest_NonLinear")


# --- Utility Functions for MLP Testing ---
def create_dummy_z(d=3, k=2, requires_grad=False, seed=None):
    """Creates a dummy Z tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(d, k, 2)
    if requires_grad:
        z.requires_grad_(True)
    return z


def create_dummy_hparams(d=3, alpha_val=1.0, beta_val=1.0, tau_val=1.0, sigma_z_val=1.0,
                         sigma_obs_noise_val=0.1, rho_val=0.05, temp_ratio_val=1.0,
                         n_grad_mc_samples_val=2, n_nongrad_mc_samples_val=2,
                         phi_prior_sigma_val=1.0):
    """Creates a dummy hparams dictionary for MLP version."""
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
        'phi_prior_sigma': phi_prior_sigma_val,  # Changed from theta_prior_sigma
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

    # X1 (node 0) is exogenous, sample from N(0,1) 
    X_data[:, 0] = torch.randn(num_samples)

    # X2 (node 1) = Theta_true[0,1] * X1 + noise
    noise_x2 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 1] = Theta_true[0, 1] * X_data[:, 0] + noise_x2

    # X3 (node 2) = Theta_true[1,2] * X2 + noise
    noise_x3 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 2] = Theta_true[1, 2] * X_data[:, 1] + noise_x3
    
    return {'x': X_data, 'G_true': G_true, 'Theta_true': Theta_true, 'y': None}


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
        'phi_prior_sigma': cfg.model_hparams.get('phi_prior_sigma_val', 1.0),  # Changed from theta_prior_sigma
        'd': d_nodes,
    }


@hydra.main(config_path="../conf", config_name="config_mlp", version_base=None)
def run_gradient_ascent_experiment_mlp(cfg: DictConfig) -> None:
    log.info("\n--- Running MLP Gradient Ascent Test with X1->X2->X3 Ground Truth (Hydra Config) ---")
    log.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. Define Constants and Generate Ground Truth Data
    D_nodes = cfg.data.d_nodes
    K_latent = cfg.particle.k_latent
    N_samples_data = cfg.data.num_samples
    synthetic_obs_noise_std = cfg.data.synthetic_obs_noise_std
    ground_truth_seed = cfg.data.ground_truth_seed
    hidden_dim = cfg.model_hparams.get('hidden_dim', 10)  # Add hidden dimension config

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    data_package = generate_ground_truth_data_x1_x2_x3(
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

    # 2. Initialize Particle (Z and Phi networks instead of Theta)
    z_init_seed = cfg.seed + cfg.particle.init_seed_offset if cfg.seed is not None else None
    phi_init_seed = cfg.seed + cfg.particle.init_seed_offset + 1 if cfg.seed is not None else None

    Z_current = create_dummy_z(D_nodes, K_latent, requires_grad=False, seed=z_init_seed)
    Phi_current = create_dummy_phi_nets(D_nodes, hidden_dim=hidden_dim, seed=phi_init_seed)

    # 3. Base Hyperparameters for DiBS
    base_hparams_config = create_model_hparams_from_cfg(cfg, D_nodes, K_latent)

    # 4. Learning Rates and Iterations from cfg
    lr_z = cfg.training.lr_z
    lr_phi = cfg.training.get('lr_phi', cfg.training.lr_theta)  # Use lr_phi or fallback to lr_theta
    num_iterations = cfg.training.num_iterations
    max_grad_norm_z = cfg.training.max_grad_norm_z
    max_grad_norm_phi = cfg.training.get('max_grad_norm_phi', cfg.training.max_grad_norm_theta)
    device = cfg.device

    log.info(f"\nInitial Z (norm): {Z_current.norm().item():.4f}")
    log.info(f"Initial Phi networks: {len(Phi_current)} networks with hidden_dim={hidden_dim}")
    log.info(f"Learning rates: lr_z={lr_z}, lr_phi={lr_phi}")
    log.info(f"Iterations: {num_iterations}")
    log.info(f"Device: {device}")

    # 5. Gradient Ascent Loop
    for t_iter in range(num_iterations):
        current_annealing_t_for_hparams = torch.tensor(float(t_iter + 1))
        Z_param = Z_current.clone().detach().requires_grad_(True)
        
        # Reset gradients for phi networks
        for net in Phi_current:
            for param in net.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        params_for_calc = {'z': Z_param, 'phi': Phi_current, 't': current_annealing_t_for_hparams}

        lj_val = float('nan')
        try:
            lj = log_joint_mlp(params_for_calc, data_dict, base_hparams_config, device=device)
            lj_val = lj.item()
        except Exception as e:
            log.error(f"Error computing log_joint_mlp at iter {t_iter+1}: {e}")
            continue

        grad_z_norm = float('nan')
        grad_phi_norm = float('nan')
        try:
            # Compute gradients using backpropagation
            lj.backward()
            
            grad_z = Z_param.grad
            if grad_z is not None:
                grad_z_norm = grad_z.norm().item()
            
            # Compute phi gradient norm
            phi_grad_norms = []
            for net in Phi_current:
                for param in net.parameters():
                    if param.grad is not None:
                        phi_grad_norms.append(param.grad.norm().item())
            
            if phi_grad_norms:
                grad_phi_norm = torch.tensor(phi_grad_norms).norm().item()
            
            # Gradient clipping
            if max_grad_norm_z > 0 and grad_z is not None:
                torch.nn.utils.clip_grad_norm_([Z_param], max_norm=max_grad_norm_z)
            if max_grad_norm_phi > 0:
                torch.nn.utils.clip_grad_norm_(Phi_current.parameters(), max_norm=max_grad_norm_phi)
                
        except Exception as e:
            log.error(f"Error computing gradients at iter {t_iter+1}: {e}")
            break

        # Update parameters
        with torch.no_grad():
            if Z_param.grad is not None:
                Z_current.add_(lr_z * Z_param.grad)
            
            # Update phi networks using their gradients
            for net in Phi_current:
                for param in net.parameters():
                    if param.grad is not None:
                        param.data.add_(lr_phi * param.grad)

        if (t_iter + 1) % 50 == 0 or t_iter == 0 or (t_iter + 1) == num_iterations:
            log.info(f"Iter {t_iter+1:3d}: Z_norm={Z_current.norm().item():.4f}, "
                  f"log_joint={lj_val:.4f}, grad_Z_norm={grad_z_norm:.4e}, grad_Phi_norm={grad_phi_norm:.4e}")

            current_hparams_for_iter = update_dibs_hparams(base_hparams_config, current_annealing_t_for_hparams.item())
            log.info(f"    Annealed: alpha={current_hparams_for_iter['alpha']:.3f}, beta={current_hparams_for_iter['beta']:.3f}, tau={current_hparams_for_iter['tau']:.3f}")
            
            hparams_for_edge_probs = {'alpha': current_hparams_for_iter['alpha'], 'd': D_nodes}
            with torch.no_grad():
                current_edge_probs = bernoulli_soft_gmat(Z_current, hparams_for_edge_probs)
            log.info(f"    Current Edge Probs (from Z, alpha={hparams_for_edge_probs['alpha']:.3f}):\n{current_edge_probs.round(decimals=2)}")

            if torch.isnan(Z_current).any():
                log.error("!!! NaN detected in Z parameters. Stopping. !!!")
                break

    log.info("\n--- MLP Single Particle Gradient Ascent (X1->X2->X3 Data with Hydra) Completed ---")
    log.info(f"Final Z (norm): {Z_current.norm().item():.4f}")

    # Final comparison
    final_t_for_hparams = torch.tensor(float(num_iterations))
    final_hparams = update_dibs_hparams(base_hparams_config, final_t_for_hparams.item())
    
    G_learned_hard = torch.zeros_like(G_true)
    try:
        alpha_for_hard_gmat = final_hparams.get('alpha', 1.0) 
        G_learned_hard = hard_gmat_particles_from_z(
            Z_current.unsqueeze(0), 
            alpha_hparam_for_scores=alpha_for_hard_gmat
        ).squeeze(0).int()
    except Exception as e:
        log.error(f"Could not generate final hard graph from Z_current: {e}")

    log.info(f"Final G_learned_hard:\n{G_learned_hard}")
    log.info(f"Ground Truth G:\n{G_true.int()}")
    
    # Test the learned MLPs on some sample data
    log.info("\n--- Testing Learned MLPs ---")
    with torch.no_grad():
        test_x = data_dict['x'][:5]  # Take first 5 samples
        soft_g = bernoulli_soft_gmat(Z_current, final_hparams)
        
        # Apply MLPs to get predictions
        pred_cols = []
        for j in range(D_nodes):
            masked_x = test_x * soft_g[:, j]
            pred_cols.append(Phi_current[j](masked_x))
        pred_mean = torch.stack(pred_cols, dim=1)
        
        log.info(f"Sample input data:\n{test_x.round(decimals=3)}")
        log.info(f"MLP predictions:\n{pred_mean.round(decimals=3)}")
        log.info(f"Residuals:\n{(test_x - pred_mean).round(decimals=3)}")

    # Debugging: Show what the MLPs actually learned
    log.info("\n--- Debugging: Graph Structure vs Predictions ---")
    with torch.no_grad():
        test_x = data_dict['x'][:5]
        soft_g = bernoulli_soft_gmat(Z_current, final_hparams)
        
        log.info(f"Learned soft graph structure:\n{soft_g.round(decimals=3)}")
        log.info(f"Ground truth should be:\n{G_true}")
        
        # Show what each MLP is actually using as input
        for j in range(D_nodes):
            masked_x = test_x * soft_g[:, j]
            log.info(f"\nFor X{j+1} prediction:")
            log.info(f"  Soft weights: {soft_g[:, j].round(decimals=3)}")
            log.info(f"  Raw input sample: {test_x[0].round(decimals=3)}")
            log.info(f"  Masked input sample: {masked_x[0].round(decimals=3)}")
            log.info(f"  MLP output: {Phi_current[j](masked_x[0:1]).item():.3f}")
        
        # Compare with ground truth predictions
        log.info(f"\n--- Ground Truth Predictions (for comparison) ---")
        gt_pred = torch.zeros_like(test_x)
        gt_pred[:, 0] = test_x[:, 0]  # X1 is exogenous
        gt_pred[:, 1] = 2.0 * test_x[:, 0]  # X2 = 2*X1 (ignoring noise)
        gt_pred[:, 2] = -1.5 * test_x[:, 1]  # X3 = -1.5*X2 (ignoring noise)
        
        log.info(f"Ground truth structure predictions:\n{gt_pred[:5].round(decimals=3)}")
        log.info(f"Actual data (with noise):\n{test_x.round(decimals=3)}")
        log.info(f"GT vs Actual residuals:\n{(test_x - gt_pred[:5]).round(decimals=3)}")

    log.info(f"\n{50*'*'}")
    log.info("MLP-based DiBS inference completed successfully!")


if __name__ == '__main__':
    run_gradient_ascent_experiment_mlp()
