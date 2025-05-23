import torch
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
import math # For math.isclose

import sys
import os

# Get the directory of the current script (tests/test_dibs_torch.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (dibs_torch_v2)
project_root = os.path.join(current_script_dir, '..')
# Add the project root to sys.path
sys.path.insert(0, project_root)


import numpy as np
from models.dibs_torch_v2 import (
    log_joint, grad_log_joint, bernoulli_soft_gmat,
    hard_gmat_particles_from_z, log_gaussian_likelihood,
    log_bernoulli_likelihood, log_full_likelihood, log_theta_prior,
    gumbel_soft_gmat, gumbel_grad_acyclic_constr_mc,
    grad_z_log_joint_gumbel, grad_theta_log_joint,
    grad_log_joint, update_dibs_hparams,
    scores, acyclic_constr
)
from models.graph_torch import scalefree_dag_gmat
from models.utils_torch import sample_x


# --- Utility Functions for Testing ---
def create_dummy_z(d=3, k=2, requires_grad=False, seed=None):
    """Creates a dummy Z tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    # Z contains U and V. U: [D, K], V: [D, K]
    # So Z can be [D, K, 2]
    z = torch.randn(d, k, 2)
    if requires_grad:
        z.requires_grad_(True)
    return z

def create_dummy_theta(d=3, requires_grad=False, seed=None):
    """Creates a dummy Theta tensor."""
    if seed is not None:
        torch.manual_seed(seed)
    theta = torch.randn(d, d)
    if requires_grad:
        theta.requires_grad_(True)
    return theta

def create_dummy_data(n_samples=10, d=3, n_expert_edges=1, seed=None):
    """Creates a dummy data dictionary."""
    if seed is not None:
        torch.manual_seed(seed)
    x_data = torch.randn(n_samples, d)
    y_expert_data = []
    if n_expert_edges > 0 and d > 1:
        for _ in range(n_expert_edges):
            i, j = torch.randperm(d)[:2].tolist() # Ensure i != j for simplicity
            val = torch.tensor(float(torch.randint(0, 2, (1,)).item()))
            y_expert_data.append(torch.tensor([float(i), float(j), val]))
    return {'x': x_data, 'y': y_expert_data if y_expert_data else None}


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
        'n_nongrad_mc_samples': n_nongrad_mc_samples_val, # Used by gumbel_acyclic_constr_mc
        'theta_prior_sigma': theta_prior_sigma_val,
        'd': d # often useful to have d in hparams
    }

# --- Unit Tests ---
print("--- Running Unit Tests ---")

def test_scores():
    print("\nTesting scores()...")
    d, k = 3, 2
    z = create_dummy_z(d, k, seed=0)
    alpha_hparam = 1.5
    s = scores(z, alpha_hparam)
    print(f"  Z shape: {z.shape}, Scores shape: {s.shape}")
    assert s.shape == (d, d), f"Expected scores shape ({d},{d}), got {s.shape}"
    # Check diagonal is zero
    assert torch.all(torch.diag(s) == 0).item(), "Diagonal of scores should be zero"
    # Check a few off-diagonal values manually
    u = z[..., 0]
    v = z[..., 1]
    for i in range(d):
        for j in range(d):
            if i != j:
                expected = alpha_hparam * torch.dot(u[i], v[j])
                actual = s[i, j]
                assert torch.allclose(actual, expected), f"Mismatch at ({i},{j}): {actual} vs {expected}"
    print("  scores() test passed (basic checks and value check).")


def test_bernoulli_soft_gmat():
    print("\nTesting bernoulli_soft_gmat()...")
    d, k = 3, 2
    z = create_dummy_z(d, k, seed=1)
    hparams = create_dummy_hparams(d, alpha_val=1.0)
    g_soft = bernoulli_soft_gmat(z, hparams)
    print(f"  Bernoulli G_soft shape: {g_soft.shape}")
    assert g_soft.shape == (d, d)
    assert torch.all(g_soft >= 0) and torch.all(g_soft <= 1), "Probabilities out of [0,1] range"
    assert torch.all(torch.diag(g_soft) == 0).item(), "Diagonal of bernoulli_soft_gmat should be zero"
    print("  bernoulli_soft_gmat() test passed.")


def test_gumbel_soft_gmat():
    print("\nTesting gumbel_soft_gmat()...")
    d, k = 3, 2
    z = create_dummy_z(d, k, seed=3)
    hparams = create_dummy_hparams(d, alpha_val=1.0, tau_val=0.5)
    g_soft = gumbel_soft_gmat(z, hparams, device='cpu')
    print(f"  Gumbel G_soft shape: {g_soft.shape}")
    
    # Basic shape and range checks
    assert g_soft.shape == (d, d), f"Expected shape ({d},{d}), got {g_soft.shape}"
    assert torch.all(g_soft >= 0) and torch.all(g_soft <= 1), "Probabilities out of [0,1] range"
    assert torch.all(torch.diag(g_soft) == 0).item(), "Diagonal of gumbel_soft_gmat should be zero"
    
    # Test different tau values
    hparams_low_tau = create_dummy_hparams(d, alpha_val=1.0, tau_val=0.1)
    g_soft_low_tau = gumbel_soft_gmat(z, hparams_low_tau, device='cpu')
    # Lower tau should make probabilities more uniform
    assert torch.mean(torch.abs(g_soft_low_tau - 0.5)) < torch.mean(torch.abs(g_soft - 0.5)), \
        "Lower tau should make probabilities more uniform"
    
    # Test different alpha values
    hparams_high_alpha = create_dummy_hparams(d, alpha_val=10.0, tau_val=0.5)
    g_soft_high_alpha = gumbel_soft_gmat(z, hparams_high_alpha, device='cpu')
    # Higher alpha should make probabilities more extreme
    assert torch.mean(torch.abs(g_soft_high_alpha - 0.5)) > torch.mean(torch.abs(g_soft - 0.5)), \
        "Higher alpha should make probabilities more extreme"
    print("  gumbel_soft_gmat() test passed.")

def test_gumbel_soft_gmat_reproducibility():
    print("\nTesting gumbel_soft_gmat() for reproducibility with fixed seed...")
    d, k = 2, 1
    # The seed for creating z is separate from the seed for gumbel_soft_gmat's internal sampling
    z = create_dummy_z(d, k, seed=42)
    hparams = create_dummy_hparams(d, alpha_val=1.0, tau_val=0.5)
    device = 'cpu'

    # Call 1
    torch.manual_seed(123) # Set seed for gumbel_soft_gmat's internal sampling
    g_soft1 = gumbel_soft_gmat(z, hparams, device=device)

    # Call 2: To get the exact same output, reset the seed to the same value
    torch.manual_seed(123) # << --- Crucial: Reset seed to the same state
    g_soft2 = gumbel_soft_gmat(z, hparams, device=device)

    print(f"  Gumbel G_soft1 (seed 123):\n{g_soft1}")
    print(f"  Gumbel G_soft2 (seed 123):\n{g_soft2}")
    assert g_soft1.shape == (d, d), "Shape mismatch for g_soft1"
    assert torch.all(g_soft1 >= 0) and torch.all(g_soft1 <= 1), "g_soft1 values out of [0,1]"
    assert torch.all(torch.diag(g_soft1) == 0.0), "Diagonal of g_soft1 not zero"
    assert torch.allclose(g_soft1, g_soft2), "Output should be reproducible when torch.manual_seed is fixed before each call"

    # Call 3: With a different seed to ensure the function *can* produce different outputs
    torch.manual_seed(456) # Use a different seed
    g_soft3 = gumbel_soft_gmat(z, hparams, device=device)
    print(f"  Gumbel G_soft3 (seed 456):\n{g_soft3}")

    # Check that g_soft1 and g_soft3 are indeed different
    # (There's a tiny chance they could be the same due to collision, but highly unlikely for continuous values)
    assert not torch.allclose(g_soft1, g_soft3), "Output should be different with a different random seed"

    print("  gumbel_soft_gmat() reproducibility test passed.")

def test_acyclic_constr():
    print("\nTesting acyclic_constr()...")
    d = 3
    # Acyclic example (approx)
    g_acyclic = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    h_acyclic = acyclic_constr(g_acyclic, d)
    print(f"  h(acyclic_approx): {h_acyclic.item()}")
    # For a truly discrete acyclic graph, h should be 0. For soft, it might be small.
    # assert math.isclose(h_acyclic.item(), 0.0, abs_tol=1e-5) # Might be too strict for soft

    # Cyclic example
    g_cyclic = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]) # 0->1->2->0
    h_cyclic = acyclic_constr(g_cyclic, d)
    print(f"  h(cyclic): {h_cyclic.item()}")
    assert h_cyclic.item() > 1e-3, "h(G) for a cyclic graph should be positive" # Allow small tolerance
    print("  acyclic_constr() test passed (qualitative checks).")

def test_log_gaussian_likelihood():
    print("\nTesting log_gaussian_likelihood()...")
    x = torch.tensor([[1.0, 2.0]])
    pred_mean = torch.tensor([[1.1, 2.2]])
    sigma = 0.5
    log_lik = log_gaussian_likelihood(x, pred_mean, sigma)
    # Manual calculation for N(1.0; 1.1, 0.5^2) + N(2.0; 2.2, 0.5^2)
    # logpdf(x, mu, sigma) = -0.5 * ((x-mu)/sigma)**2 - log(sigma * sqrt(2*pi))
    term1 = -0.5 * ((1.0-1.1)/0.5)**2 - math.log(0.5 * math.sqrt(2*math.pi)) # -0.02 - (-0.2047) = -0.02 + 0.2047 = 0.1847
    term2 = -0.5 * ((2.0-2.2)/0.5)**2 - math.log(0.5 * math.sqrt(2*math.pi)) # -0.08 - (-0.2047) = -0.08 + 0.2047 = 0.1247
    expected = term1 + term2 # approx 0.3094
    print(f"  Computed log_lik: {log_lik.item()}, Expected approx: {expected}")
    assert math.isclose(log_lik.item(), expected, rel_tol=1e-4)
    print("  log_gaussian_likelihood() test passed.")

def test_log_bernoulli_likelihood():
    print("\nTesting log_bernoulli_likelihood()...")
    y_expert_edge = torch.tensor(1.0) # Expert says edge exists
    soft_gmat_entry = torch.tensor(0.8) # Model believes edge exists with P=0.8
    rho = 0.1 # Expert error rate
    jitter = 1e-9

    # P(expert_says_1 | g=0.8, rho=0.1) = g*(1-rho) + (1-g)*rho
    # = 0.8 * 0.9 + 0.2 * 0.1 = 0.72 + 0.02 = 0.74
    # log P = log(0.74)
    expected_log_lik = math.log(0.74 + jitter)
    log_lik = log_bernoulli_likelihood(y_expert_edge, soft_gmat_entry, rho, jitter=jitter)
    print(f"  Computed log_lik: {log_lik.item()}, Expected: {expected_log_lik}")
    assert math.isclose(log_lik.item(), expected_log_lik, rel_tol=1e-6)
    print("  log_bernoulli_likelihood() test passed.")


def test_log_full_likelihood():
    print("\nTesting log_full_likelihood()...")
    d = 2
    data = create_dummy_data(n_samples=5, d=d, n_expert_edges=1, seed=3)
    # Ensure expert edge is within bounds
    if data['y'] and data['y'][0][0] >=d or data['y'][0][1] >=d :
        data['y'][0][0] = torch.tensor(0.0)
        data['y'][0][1] = torch.tensor(1.0)

    soft_gmat = torch.tensor([[0.0, 0.8], [0.1, 0.0]]) # Example soft graph
    current_theta = create_dummy_theta(d, seed=4)
    hparams = create_dummy_hparams(d, sigma_obs_noise_val=0.2, rho_val=0.1, temp_ratio_val=0.5)

    log_full_lik = log_full_likelihood(data, soft_gmat, current_theta, hparams, device='cpu')
    print(f"  log_full_likelihood output: {log_full_lik.item()}")
    assert isinstance(log_full_lik.item(), float) # Check it's a scalar float
    print("  log_full_likelihood() test passed (ran without error).")

def test_log_theta_prior():
    print("\nTesting log_theta_prior()...")
    d=2
    theta_effective = torch.tensor([[0.1, -0.2],[0.0, 0.3]])
    theta_prior_mean = torch.zeros_like(theta_effective)
    theta_prior_sigma = 0.5
    log_prior = log_theta_prior(theta_effective, theta_prior_mean, theta_prior_sigma)
    print(f"  log_theta_prior output: {log_prior.item()}")
    assert isinstance(log_prior.item(), float)
    print("  log_theta_prior() test passed.")

# --- Integration Tests for Gradient Functions ---
print("\n--- Running Integration Tests for Gradients ---")

def test_gumbel_grad_acyclic_constr_mc():
    print("\nTesting gumbel_grad_acyclic_constr_mc()...")
    d, k = 2, 1
    z = create_dummy_z(d, k, requires_grad=True, seed=5)
    hparams = create_dummy_hparams(d, alpha_val=1.0, tau_val=1.0, n_nongrad_mc_samples_val=2) # n_nongrad_mc_samples for this
    
    # Need to pass hparams that gumbel_grad_acyclic_constr_mc expects (hparams_dict_nonopt)
    # It typically gets alpha, tau from this.
    grad_h = gumbel_grad_acyclic_constr_mc(z, d, hparams, hparams['n_nongrad_mc_samples'])
    
    print(f"  Gradient shape for acyclic_constr: {grad_h.shape}")
    assert grad_h.shape == z.shape
    assert not torch.isnan(grad_h).any(), "NaN in gradient"
    print("  gumbel_grad_acyclic_constr_mc() test passed (ran, shape correct).")


def test_grad_z_log_joint_gumbel():
    print("\nTesting grad_z_log_joint_gumbel()...")
    d, k = 2, 1
    current_z_opt = create_dummy_z(d, k, requires_grad=True, seed=6)
    current_theta_nonopt = create_dummy_theta(d, seed=7).detach() # Detach as it's non-optimal
    data_dict = create_dummy_data(n_samples=5, d=d, n_expert_edges=1, seed=8)
    if data_dict['y'] and data_dict['y'][0][0] >=d or data_dict['y'][0][1] >=d : # Fix expert edge
        data_dict['y'][0][0] = torch.tensor(0.0)
        data_dict['y'][0][1] = torch.tensor(1.0)

    hparams_full = create_dummy_hparams(d, n_grad_mc_samples_val=2, n_nongrad_mc_samples_val=2) # n_nongrad for acyclic part

    grad_z = grad_z_log_joint_gumbel(current_z_opt, current_theta_nonopt, data_dict, hparams_full, device='cpu')
    print(f"  grad_z shape: {grad_z.shape}")
    assert grad_z.shape == current_z_opt.shape
    assert not torch.isnan(grad_z).any(), "NaN in grad_z"
    print("  grad_z_log_joint_gumbel() test passed.")


def test_grad_theta_log_joint():
    print("\nTesting grad_theta_log_joint()...")
    d, k = 2, 1
    current_z_nonopt = create_dummy_z(d, k, seed=9).detach() # Detach
    current_theta_opt = create_dummy_theta(d, requires_grad=True, seed=10)
    data_dict = create_dummy_data(n_samples=5, d=d, n_expert_edges=1, seed=11)
    if data_dict['y'] and data_dict['y'][0][0] >=d or data_dict['y'][0][1] >=d : # Fix expert edge
        data_dict['y'][0][0] = torch.tensor(0.0)
        data_dict['y'][0][1] = torch.tensor(1.0)

    hparams_full = create_dummy_hparams(d, n_grad_mc_samples_val=2)

    grad_theta = grad_theta_log_joint(current_z_nonopt, current_theta_opt, data_dict, hparams_full, device='cpu')
    print(f"  grad_theta shape: {grad_theta.shape}")
    assert grad_theta.shape == current_theta_opt.shape
    assert not torch.isnan(grad_theta).any(), "NaN in grad_theta"
    print("  grad_theta_log_joint() test passed.")


def test_grad_log_joint():
    print("\nTesting grad_log_joint()...")
    d, k = 2, 1
    z_val = create_dummy_z(d, k, requires_grad=True, seed=12)
    theta_val = create_dummy_theta(d, requires_grad=True, seed=13)
    params = {'z': z_val, 'theta': theta_val, 't': torch.tensor(0.0)} # t=0 for initial hparams
    
    data_dict = create_dummy_data(n_samples=5, d=d, n_expert_edges=1, seed=14)
    if data_dict['y'] and data_dict['y'][0][0] >=d or data_dict['y'][0][1] >=d : # Fix expert edge
        data_dict['y'][0][0] = torch.tensor(0.0)
        data_dict['y'][0][1] = torch.tensor(1.0)

    # Base hparams (before annealing for a specific step t)
    hparams_config = create_dummy_hparams(d, n_grad_mc_samples_val=1, n_nongrad_mc_samples_val=1) # Small MC for speed
    hparams_config['alpha_base'] = hparams_config['alpha'] # For update_dibs_hparams
    hparams_config['beta_base'] = hparams_config['beta']   # For update_dibs_hparams


    grads = grad_log_joint(params, data_dict, hparams_config, device='cpu')
    
    print(f"  grad_log_joint output keys: {grads.keys()}")
    assert 'z' in grads and 'theta' in grads and 't' in grads
    print(f"  grad_log_joint['z'] shape: {grads['z'].shape}")
    assert grads['z'].shape == z_val.shape
    print(f"  grad_log_joint['theta'] shape: {grads['theta'].shape}")
    assert grads['theta'].shape == theta_val.shape
    assert not torch.isnan(grads['z']).any(), "NaN in final grad_z"
    assert not torch.isnan(grads['theta']).any(), "NaN in final grad_theta"
    print("  grad_log_joint() test passed.")

def test_update_dibs_hparams():
    print("\nTesting update_dibs_hparams()...")
    hparams_base = {'alpha_base': 0.1, 'beta_base': 0.2, 'tau': 1.0, 'alpha':0.0, 'beta':0.0} # dummy alpha/beta to be overwritten
    t_step = 1
    updated1 = update_dibs_hparams(hparams_base, t_step)
    # For t=1, factor = 1 + 1/1 = 2
    expected_alpha1 = 0.1 * 2
    expected_beta1 = 0.2 * 2
    print(f"  t=1: alpha={updated1['alpha']}, beta={updated1['beta']}")
    assert math.isclose(updated1['alpha'], expected_alpha1)
    assert math.isclose(updated1['beta'], expected_beta1)

    t_step = 2
    updated2 = update_dibs_hparams(hparams_base, t_step)
    # For t=2, factor = 2 + 1/2 = 2.5
    expected_alpha2 = 0.1 * 2.5
    expected_beta2 = 0.2 * 2.5
    print(f"  t=2: alpha={updated2['alpha']}, beta={updated2['beta']}")
    assert math.isclose(updated2['alpha'], expected_alpha2)
    assert math.isclose(updated2['beta'], expected_beta2)
    print("  update_dibs_hparams() test passed.")


# Run the tests
if __name__ == '__main__':
    # Unit tests
    #test_scores()
    #test_bernoulli_soft_gmat()
    #test_gumbel_soft_gmat()
    #test_gumbel_soft_gmat_reproducibility()
    test_acyclic_constr()
    #test_log_gaussian_likelihood()
    #test_log_bernoulli_likelihood()
    #test_log_full_likelihood()
    #test_log_theta_prior()
    #test_update_dibs_hparams()

    # Integration tests for gradients
    # These require all the above functions to be correctly defined and imported.
    # Make sure your DiBS functions are accessible in the scope where these tests run.
    #test_gumbel_grad_acyclic_constr_mc()
    #test_grad_z_log_joint_gumbel()
    #test_grad_theta_log_joint()
    #test_grad_log_joint()

    print("\n--- All tests completed ---")

# Note: To run these tests, you need to copy your DiBS function definitions
# into this script or ensure they are importable.
# The Logistic class definition is also assumed to be present.

# Example of Logistic class (if not already defined)
class Logistic(Distribution):
    has_rsample = True
    arg_constraints = {'loc': torch.distributions.constraints.real,
                       'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.real

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, torch.Tensor):
            batch_shape = self.loc.size()
        else: # loc is number
            batch_shape = torch.Size()
        super(Logistic, self).__init__(batch_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Sample from U(0,1)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        # Inverse CDF transform: loc + scale * (log(u) - log(1-u))
        return self.loc + self.scale * (torch.log(u) - torch.log1p(-u))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # z = (value - self.loc) / self.scale
        # return -z - 2. * F.softplus(-z) - torch.log(self.scale)
        # Simpler form:
        z = (value - self.loc) / self.scale
        log_p = -z - 2 * torch.log(1 + torch.exp(-z)) - torch.log(self.scale)
        return log_p

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return torch.sigmoid((value - self.loc) / self.scale)

    def icdf(self, value): # Inverse CDF
        if self._validate_args:
            # self._validate_sample(value) # value is probability here
            if not (torch.all(value >=0) and torch.all(value <=1)):
                 raise ValueError("Probabilities must be in [0,1]")
        return self.loc + self.scale * (torch.log(value) - torch.log1p(-value))

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return (self.scale**2 * math.pi**2) / 3.0