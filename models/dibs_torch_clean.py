import math
import logging
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all

__all__ = [
    "acyclic_constr",
    "stable_mean",
    "bernoulli_soft_gmat",
    "bernoulli_sample_gmat",
    "gumbel_soft_gmat",
    "log_full_likelihood",
    "gumbel_acyclic_constr_mc",
    "grad_z_log_joint_gumbel",
    "grad_theta_log_joint",
    "grad_log_joint",
    "log_joint",
    "update_dibs_hparams",
    "hard_gmat_particles_from_z",
    "Logistic",
]

log = logging.getLogger("DiBS")

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def acyclic_constr(g: torch.Tensor, d: int) -> torch.Tensor:
    """H(G) from NOTEARS (Zheng et al.) with a series fallback for large *d*."""
    alpha = 1.0 / d
    eye = torch.eye(d, device=g.device, dtype=g.dtype)
    m = eye + alpha * g

    if d <= 10:
        return torch.trace(torch.linalg.matrix_power(m, d)) - d

    try:
        eigvals = torch.linalg.eigvals(m)
        return torch.sum(torch.real(eigvals ** d)) - d
    except RuntimeError:
        trace, p = torch.tensor(0.0, device=g.device, dtype=g.dtype), g.clone()
        for k in range(1, min(d + 1, 20)):
            trace += (alpha ** k) * torch.trace(p) / k
            if k < 19:
                p = p @ g
        return trace


def stable_mean(x: torch.Tensor, dim: int = 0, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable mean for tensors spanning many orders of magnitude."""
    jitter = 1e-30
    if not x.is_floating_point():
        x = x.float()

    pos, neg = x.clamp(min=0), (-x).clamp(min=0)
    sum_pos = torch.exp(torch.logsumexp(torch.log(pos + jitter), dim=dim, keepdim=True))
    sum_neg = torch.exp(torch.logsumexp(torch.log(neg + jitter), dim=dim, keepdim=True))

    n = torch.tensor(x.shape[dim] if dim is not None else x.numel(), dtype=x.dtype, device=x.device)
    mean = (sum_pos - sum_neg) / (n + jitter)
    return mean if keepdim else mean.squeeze(dim)


# -----------------------------------------------------------------------------
# Graph parameterisations
# -----------------------------------------------------------------------------

def _scores(z: torch.Tensor, alpha: float) -> torch.Tensor:
    u, v = z[..., 0], z[..., 1]  # [..., D, K]
    raw = alpha * torch.einsum("...ik,...jk->...ij", u, v)
    d = raw.shape[-1]
    mask = 1.0 - torch.eye(d, device=z.device, dtype=z.dtype)
    if raw.ndim == 3:
        mask = mask.expand(raw.shape[0], d, d)
    return raw * mask


def bernoulli_soft_gmat(z: torch.Tensor, h: Dict[str, Any]) -> torch.Tensor:
    """Edge‑probability matrix σ(α·uᵀv) with zero diagonal."""
    p = torch.sigmoid(_scores(z, h["alpha"]))
    d = p.shape[-1]
    diag_mask = 1.0 - torch.eye(d, device=p.device, dtype=p.dtype)
    return p if p.ndim == 2 else p * diag_mask


def bernoulli_sample_gmat(z: torch.Tensor, h: Dict[str, Any]) -> torch.Tensor:
    return torch.bernoulli(bernoulli_soft_gmat(z, h))


def gumbel_soft_gmat(z: torch.Tensor, h: Dict[str, Any]) -> torch.Tensor:
    """Concrete distribution re‑parameterisation (Gumbel‑sigmoid)."""
    raw = _scores(z, h["alpha"])
    u = torch.rand_like(raw)
    noise = torch.log(u) - torch.log1p(-u)
    soft = torch.sigmoid((raw + noise) * h["tau"])
    d = soft.shape[-1]
    mask = 1.0 - torch.eye(d, device=soft.device, dtype=soft.dtype)
    return soft * mask

# -----------------------------------------------------------------------------
# Likelihood
# -----------------------------------------------------------------------------

def _gauss_ll(x: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = torch.clamp(torch.tensor(sigma, dtype=x.dtype, device=x.device), 1e-6, 1e3)
    norm = -0.5 * math.log(2 * math.pi) - torch.log(sigma)
    return (norm - 0.5 * ((x - mu) / sigma) ** 2).mean()


def _expert_ll(y: torch.Tensor, p: torch.Tensor, rho: float) -> torch.Tensor:
    p1 = p * (1 - rho) + (1 - p) * rho
    p0 = p * rho + (1 - p) * (1 - rho)
    return (y * torch.log(p1) + (1 - y) * torch.log(p0)).sum()


def log_full_likelihood(data: Dict[str, Any], g_soft: torch.Tensor, theta: torch.Tensor, h: Dict[str, Any]) -> torch.Tensor:
    x = data["x"]
    mu = x @ (theta * g_soft)
    ll = _gauss_ll(x, mu, h.get("sigma_obs_noise", 0.1))

    if data.get("y") is not None:
        edges = data["y"]  # list of (i,j,val)
        e_ll = 0.0
        for i, j, val in edges:
            e_ll += _expert_ll(torch.tensor(val, device=x.device), g_soft[i, j], h["rho"])
        ll += h.get("temp_ratio", 0.0) * e_ll
    return ll

# -----------------------------------------------------------------------------
# Priors and utility
# -----------------------------------------------------------------------------

def gumbel_acyclic_constr_mc(z: torch.Tensor, d: int, h: Dict[str, Any]) -> torch.Tensor:
    vals = [acyclic_constr(torch.bernoulli(gumbel_soft_gmat(z, h)), d) for _ in range(h["n_nongrad_mc_samples"])]
    return torch.stack(vals).mean()


def _theta_prior(theta_eff: torch.Tensor, sigma: float) -> torch.Tensor:
    return _gauss_ll(theta_eff, torch.zeros_like(theta_eff), sigma)

# -----------------------------------------------------------------------------
# Gradients
# -----------------------------------------------------------------------------

def grad_z_log_joint_gumbel(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], h: Dict[str, Any]) -> torch.Tensor:
    d = z.shape[0]
    beta = h["beta"]
    sigma_z2 = h["sigma_z"] ** 2

    # prior gradient
    grad_h = torch.autograd.grad(
        acyclic_constr(gumbel_soft_gmat(z, h), d), z, retain_graph=True
    )[0]
    grad_prior = -beta * grad_h - z / sigma_z2

    # likelihood gradient (REINFORCE style, low‑variance version)
    gs = gumbel_soft_gmat(z, h)
    ll = log_full_likelihood(data, gs, theta, h)
    grad_ll = torch.autograd.grad(ll, z)[0]
    return grad_prior + grad_ll


def grad_theta_log_joint(z: torch.Tensor, theta: torch.Tensor, data: Dict[str, Any], h: Dict[str, Any]) -> torch.Tensor:
    g_soft = bernoulli_soft_gmat(z, h)
    ll = log_full_likelihood(data, g_soft, theta, h)
    prior = _theta_prior(theta * g_soft, h["theta_prior_sigma"])
    return torch.autograd.grad(ll + prior, theta)[0]

# -----------------------------------------------------------------------------
# Top‑level interfaces
# -----------------------------------------------------------------------------

def grad_log_joint(params: Dict[str, Any], data: Dict[str, Any], h_cfg: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    h = update_dibs_hparams(h_cfg, params["t"].item())
    dz = grad_z_log_joint_gumbel(params["z"], params["theta"].detach(), data, h)
    dtheta = grad_theta_log_joint(params["z"].detach(), params["theta"], data, h)
    return {"z": dz, "theta": dtheta, "t": torch.zeros(1, device=dz.device)}


def log_joint(params: Dict[str, Any], data: Dict[str, Any], h_cfg: Dict[str, Any]) -> torch.Tensor:
    h = update_dibs_hparams(h_cfg, params["t"].item())
    z, theta = params["z"], params["theta"]
    d = z.shape[0]

    g_soft = bernoulli_soft_gmat(z, h)
    ll = log_full_likelihood(data, g_soft, theta, h)
    prior_z = Normal(0.0, h["sigma_z"]).log_prob(z).sum() - h["beta"] * gumbel_acyclic_constr_mc(z, d, h)
    prior_theta = _theta_prior(theta * g_soft, h["theta_prior_sigma"])
    return ll + prior_z + prior_theta


def update_dibs_hparams(h: Dict[str, Any], t: float) -> Dict[str, Any]:
    out = dict(h)
    fac = t + 1.0 / max(t, 1e-3)
    out["beta"] = h["beta"] * fac
    return out


def hard_gmat_particles_from_z(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    s = _scores(z, alpha)
    return (s > 0).float()


class Logistic(Distribution):
    """Simple logistic distribution (PyTorch doesn’t ship one)."""

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        super().__init__(self.loc.size(), validate_args)

    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(self._extended_shape(sample_shape), device=self.loc.device, dtype=self.loc.dtype)
        return self.loc + self.scale * (torch.log(u) - torch.log1p(-u))

    def log_prob(self, value):
        z = (value - self.loc) / self.scale
        return -z - 2 * torch.log1p(torch.exp(-z)) - torch.log(self.scale)
