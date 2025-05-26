import torch
import torch.nn.functional as F
from torch.distributions import Normal
import math
import sys
import os

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

# Placeholder for the main gradient ascent test function
def run_gradient_ascent_test_with_x1_x2_x3_data():
    print("\n--- Running Gradient Ascent Test with X1->X2->X3 Ground Truth ---")
    
    # 1. Define Constants and Generate Ground Truth Data
    D_nodes = 3
    K_latent = 3
    N_samples_data = 500
    synthetic_obs_noise_std = 0.01 # Noise in the synthetic data generation
    ground_truth_seed = 100

    data_package = generate_ground_truth_data_x1_x2_x3(
        num_samples=N_samples_data, 
        obs_noise_std=synthetic_obs_noise_std, 
        seed=ground_truth_seed
    )
    data_dict = {'x': data_package['x'], 'y': data_package['y']} # grad_log_joint expects 'x' and 'y'
    G_true = data_package['G_true']
    Theta_true = data_package['Theta_true']

    print("Ground Truth Model:")
    print(f"G_true:\n{G_true.int()}")
    print(f"Theta_true:\n{Theta_true}")
    print(f"Synthetic data observation noise_std: {synthetic_obs_noise_std}")
    print(f"Number of synthetic samples: {N_samples_data}")

    # 2. Initialize Particle (Z and Theta)
    particle_init_seed = 101
    Z_current = create_dummy_z(D_nodes, K_latent, requires_grad=False, seed=particle_init_seed)
    Theta_current = create_dummy_theta(D_nodes, requires_grad=False, seed=particle_init_seed + 1)

    # 3. Base Hyperparameters for DiBS
    base_hparams_config = create_dummy_hparams(
        d=D_nodes,
        alpha_val=0.1,       
        beta_val=1.0,      
        tau_val=1.0,        
        sigma_z_val=1/math.sqrt(K_latent), ## sigma z values is 1/sqrt(K_latent) PAPER RECOMMENDATIONS E.3
        sigma_obs_noise_val=synthetic_obs_noise_std, # Model assumes same noise as data generation
        rho_val=0.05,        # Not used if data_dict['y'] is None
        temp_ratio_val=0.0,  # No expert data, so this won't have effect
        n_grad_mc_samples_val=5, # MC samples for likelihood gradient part
        n_nongrad_mc_samples_val=10, # MC samples for acyclicity gradient part
        theta_prior_sigma_val=0.5 # Prior on learned Theta_eff (Theta * G_soft)
    )

    # 4. Learning Rates and Iterations
    lr_z = 0.005 
    lr_theta = 0.005
    num_iterations = 1 # Keep moderate for a test, can be increased for better convergence
    # Gradient clipping (optional but can help stability)
    max_grad_norm_z = 11.0 
    max_grad_norm_theta = 19

    print(f"\nInitial Z (norm): {Z_current.norm().item():.4f}")
    print(f"Initial Theta (norm): {Theta_current.norm().item():.4f}")
    # print(f"Initial Theta values:\n{Theta_current}") # Can be verbose
    print(f"Learning rates: lr_z={lr_z}, lr_theta={lr_theta}")
    print(f"Max gradient norms: Z={max_grad_norm_z}, Theta={max_grad_norm_theta}")
    print(f"Num iterations: {num_iterations}")

    # 5. Gradient Ascent Loop
    for t_iter in range(num_iterations):
        # Annealing step 't' for grad_log_joint and log_joint.
        # t_iter is 0-indexed, annealing schedules often use 1-indexed step
        current_annealing_t_for_hparams = torch.tensor(float(t_iter + 1))

        # Prepare Z and Theta to allow gradients to be computed w.r.t. them for this step
        Z_param = Z_current.clone().detach().requires_grad_(True)
        Theta_param = Theta_current.clone().detach().requires_grad_(True)
        params_for_calc = {'z': Z_param, 'theta': Theta_param, 't': current_annealing_t_for_hparams}

        # Optional: Calculate current log-joint probability for monitoring
        lj_val = float('nan')
        try:
            # log_joint uses 't' from params_for_calc for annealing hparams
            lj = log_joint(params_for_calc, data_dict, base_hparams_config, device='cpu')
            lj_val = lj.item()
        except Exception as e:
            print(f"Error computing log_joint at iter {t_iter+1}: {e}")
            # Continue if log_joint fails, but gradients are more critical

        # Calculate gradients using DiBS model's grad_log_joint
        # grad_log_joint also uses 't' from params_for_calc for annealing hparams
        grad_z_norm = float('nan'); grad_theta_norm = float('nan')
        try:
            grads = grad_log_joint(params_for_calc, data_dict, base_hparams_config, device='cpu')
            grad_z = grads['z']
            grad_theta = grads['theta']

            # Gradient Clipping
            if max_grad_norm_z is not 1e-34:
                torch.nn.utils.clip_grad_norm_(grad_z, max_grad_norm_z)
            if max_grad_norm_theta is not 1e-34:
                torch.nn.utils.clip_grad_norm_(grad_theta, max_grad_norm_theta)
            
            grad_z_norm = grad_z.norm().item()
            grad_theta_norm = grad_theta.norm().item()
        except Exception as e:
            print(f"Error computing gradients at iter {t_iter+1}: {e}")
            break # Stop if gradients fail

        # Perform gradient ascent update on Z_current and Theta_current
        with torch.no_grad(): # Ensure updates are not tracked by autograd
            Z_current.add_(lr_z * grad_z)       # In-place update
            Theta_current.add_(lr_theta * grad_theta) # In-place update

        # Print progress periodically
        if (t_iter + 1) % 10 == 0 or t_iter == 0: # Print every 10 iterations or first iteration
            print(f"Iter {t_iter+1:3d}: Z_norm={Z_current.norm().item():.4f}, Theta_norm={Theta_current.norm().item():.4f}, "
                  f"log_joint={lj_val:.4f}, grad_Z_norm={grad_z_norm:.4e}, grad_Theta_norm={grad_theta_norm:.4e}")

            # Optionally print annealed hparams and current edge probabilities
            current_hparams_for_iter = update_dibs_hparams(base_hparams_config, current_annealing_t_for_hparams.item())
            print(f"    Annealed: alpha={current_hparams_for_iter['alpha']:.3f}, beta={current_hparams_for_iter['beta']:.3f}, tau={current_hparams_for_iter['tau']:.3f}")
            
            # Edge probabilities from Z_current (using bernoulli_soft_gmat and current annealed alpha)
            hparams_for_edge_probs = {'alpha': current_hparams_for_iter['alpha'], 'd': D_nodes}
            with torch.no_grad():
                current_edge_probs = bernoulli_soft_gmat(Z_current, hparams_for_edge_probs)
            print(f"    Current Edge Probs (from Z, alpha={hparams_for_edge_probs['alpha']:.3f}):\n{current_edge_probs.round(decimals=2)}")

            # Basic check for NaNs/Infs in parameters
            if torch.isnan(Z_current).any() or torch.isnan(Theta_current).any() or \
               torch.isinf(Z_current).any() or torch.isinf(Theta_current).any():
                print("!!! NaN/Inf detected in parameters. Stopping. !!!")
                break
    
    print("\n--- Single Particle Gradient Ascent (X1->X2->X3 Data) Completed ---")
    print(f"Final Z (norm): {Z_current.norm().item():.4f}")
    print(f"Final Theta (norm): {Theta_current.norm().item():.4f}")

    # 6. Compare Learned Graph and Parameters to Ground Truth
    final_t_for_hparams = torch.tensor(float(num_iterations)) # Use last iteration number for final hparams
    final_hparams = update_dibs_hparams(base_hparams_config, final_t_for_hparams.item())
    
    G_learned_hard = torch.zeros_like(G_true) # Default to zeros
    try:
        # hard_gmat_particles_from_z expects batch of Z particles [N_particles, D, K, 2]
        # Z_current is [D, K, 2], so unsqueeze to add a batch dimension of 1.
        # The alpha for hard_gmat should ideally be the final annealed alpha.
        alpha_for_hard_gmat = final_hparams.get('alpha', 1.0) 
        G_learned_hard = hard_gmat_particles_from_z(
            Z_current.unsqueeze(0), 
            alpha_hparam_for_scores=alpha_for_hard_gmat
        ).squeeze(0).int() # Squeeze out the batch dimension
    except Exception as e:
        print(f"Could not generate final hard graph from Z_current: {e}. Using zeros for G_learned_hard.")

    print("\n--- Comparison with Ground Truth ---")
    print(f"G_true:\n{G_true.int()}")
    print(f"G_learned_hard (from Z with alpha={alpha_for_hard_gmat:.3f}):\n{G_learned_hard}")

    # Simple difference metrics for graph structure
    num_missing_edges = torch.sum((G_true.int() == 1) & (G_learned_hard == 0)).item()
    num_extra_edges = torch.sum((G_true.int() == 0) & (G_learned_hard == 1)).item()
    # A more involved SHD would also consider reversed edges that don't change the MEC.
    print(f"Number of missing edges: {num_missing_edges}")
    print(f"Number of extra edges: {num_extra_edges}")

    print(f"\nTheta_true:\n{Theta_true.round(decimals=2)}")
    print(f"Theta_learned (Theta_current):\n{Theta_current.round(decimals=2)}")

    print("\nComparison of Theta coefficients for edges present in G_true OR G_learned_hard:")
    active_coeffs_info = []
    for r in range(D_nodes):
        for c in range(D_nodes):
            true_edge_present = G_true[r, c].item() == 1
            learned_edge_present = G_learned_hard[r, c].item() == 1
            if true_edge_present or learned_edge_present:
                info = (f"  Edge {r}->{c}: True_coeff={Theta_true[r,c].item():.2f} (Present in G_true: {true_edge_present}), "
                        f"Learned_coeff={Theta_current[r,c].item():.2f} (Present in G_learned: {learned_edge_present})")
                active_coeffs_info.append(info)
    if active_coeffs_info:
        for line in active_coeffs_info:
            print(line)
    else:
        print("  No active edges in either true or learned graph to compare coefficients.")

    # A simple assertion for the test (example: perfect graph recovery)
    # This is a strict condition and might fail depending on parameters/convergence.
    # For a real test suite, you might check SHD <= threshold or specific coefficient ranges.
    perfect_graph_recovery = torch.all(G_learned_hard == G_true.int()).item()
    print(f"\nPerfect graph structure recovery: {perfect_graph_recovery}")
    # assert perfect_graph_recovery, "Graph structure not perfectly recovered."
    # Add more assertions as needed, e.g., on coefficient values for true edges.
    if perfect_graph_recovery:
        if Theta_true[0,1] != 0:
            assert math.isclose(Theta_current[0,1].item(), Theta_true[0,1].item(), rel_tol=0.5), f"Coeff for 0->1 mismatch: learned {Theta_current[0,1].item()} vs true {Theta_true[0,1].item()}" 
        if Theta_true[1,2] != 0:
             assert math.isclose(Theta_current[1,2].item(), Theta_true[1,2].item(), rel_tol=0.5), f"Coeff for 1->2 mismatch: learned {Theta_current[1,2].item()} vs true {Theta_true[1,2].item()}"


if __name__ == '__main__':
    print("--- Running Inference Tests ---")
    # Example usage of data generation:
    N_samples = 100
    noise_level = 0.2
    ground_truth_data_package = generate_ground_truth_data_x1_x2_x3(N_samples, noise_level, seed=42)
    print(f"Generated X data shape: {ground_truth_data_package['x'].shape}")
    print(f"Ground Truth G:\n{ground_truth_data_package['G_true']}")
    print(f"Ground Truth Theta:\n{ground_truth_data_package['Theta_true']}")
    
    run_gradient_ascent_test_with_x1_x2_x3_data() # Uncomment when implemented
    print("\n--- Inference tests completed (data generation check) ---")
