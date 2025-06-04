#!/usr/bin/env python3
"""
Simple, self‑contained DiBS **gradient‑ascent** demo (single particle) with all
hyper‑parameters collected in one place – _no_ Hydra required.

The defaults match the YAML block you posted:

* 3‑node chain ground truth (X1 → X2 → X3)
* 500 data samples, observation noise 0.1
* Latent dimension K = 3
* α = 0.1, β = 1.0, τ = 1.0, ρ = 0.05
* MC samples: 5 (likelihood) / 10 (acyclic prior)
* σ_z = 1 / √K (can scale by a divisor)
* θ prior σ = 0.5
* Learning rates: 0.005 for both Z and Θ
* Iterations: 300

Run:
    python tests/test_dibs_simple.py  \
        --num_iterations 300 --num_samples 500 --lr_z 0.005 --lr_theta 0.005
All parameters have CLI flags so you can override on the fly.
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# Project‑root import hack so we can do `import models.dibs_torch_v2`
# -----------------------------------------------------------------------------
this_file = Path(__file__).resolve()
project_root = this_file.parent.parent  # tests/ -> repo root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.dibs_torch_v2 import (
    log_joint,
    grad_log_joint,
    bernoulli_soft_gmat,
    hard_gmat_particles_from_z,
)

# -----------------------------------------------------------------------------
# Synthetic 3‑node chain generator
# -----------------------------------------------------------------------------

def generate_chain_data(n: int, noise_std: float, seed: int = 0):
    torch.manual_seed(seed)
    D = 3
    G_true = torch.zeros(D, D)
    G_true[0, 1] = 1.0
    G_true[1, 2] = 1.0

    Theta_true = torch.zeros(D, D)
    Theta_true[0, 1] = 2.0
    Theta_true[1, 2] = -1.5

    X = torch.zeros(n, D)
    X[:, 0] = torch.randn(n)
    X[:, 1] = 2.0 * X[:, 0] + torch.randn(n) * noise_std
    X[:, 2] = -1.5 * X[:, 1] + torch.randn(n) * noise_std

    return {"x": X, "y": None}, G_true, Theta_true

# -----------------------------------------------------------------------------
# Helpers to init Z and Θ
# -----------------------------------------------------------------------------

def init_z(d: int, k: int, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(d, k, 2)

def init_theta(d: int, seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(d, d)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DiBS single‑particle gradient‑ascent demo")

    # Top‑level experiment hyper‑params (defaults from your YAML)
    parser.add_argument("--num_samples", type=int, default=500, help="number of synthetic data points")
    parser.add_argument("--noise_std", type=float, default=0.1, help="observation noise σ")
    parser.add_argument("--k_latent", type=int, default=3, help="latent dimension K for Z")

    # Training params
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--lr_z", type=float, default=0.005)
    parser.add_argument("--lr_theta", type=float, default=0.005)

    # MC samples
    parser.add_argument("--n_grad_mc", type=int, default=5)
    parser.add_argument("--n_nongrad_mc", type=int, default=10)

    # DiBS core hparams
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--temp_ratio", type=float, default=0.0)
    parser.add_argument("--theta_prior_sigma", type=float, default=0.5)
    parser.add_argument("--sigma_z_divisor", type=float, default=1.0, help="σ_z = 1/√K ÷ this value")

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 1. Data ------------------------------------------------------------------
    data_dict, G_true, Theta_true = generate_chain_data(
        n=args.num_samples, noise_std=args.noise_std, seed=args.seed
    )
    data_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in data_dict.items()}
    G_true, Theta_true = G_true.to(device), Theta_true.to(device)

    # 2. Parameters & hparams --------------------------------------------------
    D = 3  # fixed for chain example
    K = args.k_latent

    Z = init_z(D, K, seed=args.seed + 1).to(device)
    Theta = init_theta(D, seed=args.seed + 2).to(device)

    sigma_z = (1.0 / math.sqrt(K)) / args.sigma_z_divisor

    hparams = {
        "alpha": args.alpha,
        "beta": args.beta,
        "tau": args.tau,
        "sigma_z": sigma_z,
        "sigma_obs_noise": args.noise_std,
        "rho": args.rho,
        "temp_ratio": args.temp_ratio,
        "n_grad_mc_samples": args.n_grad_mc,
        "n_nongrad_mc_samples": args.n_nongrad_mc,
        "theta_prior_sigma": args.theta_prior_sigma,
        "d": D,
    }

    print(f"Initial ‖Z‖ = {Z.norm():.4f}    ‖Θ‖ = {Theta.norm():.4f}\n")

    # 3. Gradient‑ascent -------------------------------------------------------
    for t in range(1, args.num_iterations + 1):
        Z.requires_grad_(True)
        Theta.requires_grad_(True)
        params = {"z": Z, "theta": Theta, "t": torch.tensor(float(t), device=device)}

        lj_val = log_joint(params, data_dict, hparams, device=device).item()
        grads = grad_log_joint(params, data_dict, hparams, device=device)

        with torch.no_grad():
            Z += args.lr_z * grads["z"]
            Theta += args.lr_theta * grads["theta"]

        if t == 1 or t % 50 == 0 or t == args.num_iterations:
            edge_probs = bernoulli_soft_gmat(Z, {"alpha": hparams["alpha"], "d": D})
            print(
                f"[{t:03d}] log_joint = {lj_val: .3f}    ‖Z‖ = {Z.norm():.2f}    ‖Θ‖ = {Theta.norm():.2f}"
            )
            print("       edge‑probs (no rounding):\n", edge_probs.detach().cpu())

    # 4. Final hard graph ------------------------------------------------------
    G_learned = hard_gmat_particles_from_z(Z.unsqueeze(0), hparams["alpha"]).squeeze(0).int()

    print("\n=== Final comparison ===")
    print("G_true:\n", G_true.int())
    print("G_learned:\n", G_learned)
    print("Θ_true (masked):\n", Theta_true)
    print("Θ_learned (masked):\n", (Theta * G_learned).round(decimals=4))

if __name__ == "__main__":
    main()
