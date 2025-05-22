import torch
import numpy as np
from ..models.dibs_torch_v2 import (
    log_joint, grad_log_joint, bernoulli_soft_gmat,
    hard_gmat_particles_from_z
)
from ..models.graph_torch import scalefree_dag_gmat
from ..models.utils_torch import sample_x

def test_dibs_synthetic():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    d = 5  # number of nodes
    n_samples = 1000  # number of data samples
    n_particles = 10  # number of SVGD particles
    n_steps = 100  # number of optimization steps
    learning_rate = 0.01
    
    # Generate a synthetic DAG
    true_gmat = scalefree_dag_gmat(d, n_edges_per_node=2, seed=42)
    print("True graph adjacency matrix:")
    print(true_gmat)
    
    # Generate synthetic data
    # For simplicity, we'll use a linear model with random coefficients
    theta_true = torch.randn(d, d) * 0.5  # random coefficients
    theta_true = theta_true * true_gmat  # mask with true graph
    
    # Sample data using the true graph and coefficients
    hparams = {
        "noise_std": torch.ones(d) * 0.1,
        "parent_means": torch.zeros(d),
        "n_scm_samples": n_samples
    }
    
    x_data = sample_x(true_gmat, theta_true, n_samples, hparams)
    print(f"\nGenerated data shape: {x_data.shape}")
    
    # Initialize DiBS particles
    k = 2  # latent dimension for Z
    z_particles = torch.randn(n_particles, d, k, 2) * 0.1
    theta_particles = torch.randn(n_particles, d, d) * 0.1
    
    # DiBS hyperparameters
    dibs_hparams = {
        "alpha": 1.0,  # score scaling
        "beta": 1.0,   # acyclicity constraint weight
        "tau": 1.0,    # Gumbel-softmax temperature
        "sigma_z": 1.0,  # prior std for Z
        "theta_prior_sigma": 1.0,  # prior std for theta
        "n_grad_mc_samples": 1,  # MC samples for gradients
        "n_nongrad_mc_samples": 1,  # MC samples for non-gradient terms
        "sigma_obs_noise": 0.1  # observation noise
    }
    
    # Prepare data dictionary
    data_dict = {"x": x_data}
    
    # Optimization loop
    print("\nStarting DiBS optimization...")
    for step in range(n_steps):
        # Update each particle
        for p in range(n_particles):
            # Current particle parameters
            params = {
                "z": z_particles[p],
                "theta": theta_particles[p],
                "t": torch.tensor([step], dtype=torch.float32)
            }
            
            # Compute gradients
            grads = grad_log_joint(params, data_dict, dibs_hparams)
            
            # Update parameters
            z_particles[p] = z_particles[p] + learning_rate * grads["z"]
            theta_particles[p] = theta_particles[p] + learning_rate * grads["theta"]
            
            # Compute log joint for monitoring
            log_joint_val = log_joint(params, data_dict, dibs_hparams)
            
            if step % 10 == 0 and p == 0:  # Print for first particle every 10 steps
                print(f"Step {step}, Particle {p}, Log joint: {log_joint_val:.2f}")
    
    # Get final hard graphs from particles
    final_hard_gmats = hard_gmat_particles_from_z(z_particles)
    
    # Compute average graph across particles
    avg_graph = torch.mean(final_hard_gmats, dim=0)
    print("\nAverage recovered graph (thresholded at 0.5):")
    print((avg_graph > 0.5).float())
    
    # Compare with true graph
    accuracy = torch.mean((avg_graph > 0.5).float() == true_gmat).item()
    print(f"\nGraph recovery accuracy: {accuracy:.2f}")
    
    return accuracy

if __name__ == "__main__":
    test_dibs_synthetic() 