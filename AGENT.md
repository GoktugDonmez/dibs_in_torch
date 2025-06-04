0 . Purpose & high-level picture
This repo is an experimental PyTorch re-implementation of DiBS: Differentiable Bayesian Structure Learning for differentiable Bayesian causal discovery.
The JAX reference code (~4 k LOC, mixed model + inference) is flattened here into two Python files so it is easier to iterate, debug, and eventually grow into a modular library:

file	role	key objects / functions
models/dibs_torch_v2.py	Core model + maths (≈1.1 k LOC)	scores, gumbel_soft_gmat, acyclic_constr, log_full_likelihood, log_joint, grad_log_joint …
tests/test_inference.py	Smoke-test & playground (≈350 LOC)	Generates a 3-node chain X1→X2→X3, runs plain gradient ascent for 100 steps, logs diagnostics via Hydra

Everything else (Hydra config files, utils, notebooks) is scaffolding.
SVGD, multi-particle inference and advanced DAG-penalty variants are not yet ported – the current loop is deliberately simple so we can validate gradients, clipping, annealing, etc.

1 . How the code hangs together
text
Copy
Edit
[gradient loop in test_inference.py]
└── builds params  { z, θ, t }
    │
    ├── log_joint()        (forward value)
    │     ├─ log_full_likelihood()
    │     ├─ log_theta_prior()
    │     └─ gumbel_acyclic_constr_mc()
    │          └─ gumbel_soft_gmat() → acyclic_constr()
    │
    └── grad_log_joint()   (top-level autograd)
          ├─ grad_z_log_joint_gumbel()
          │     ├─ gumbel_grad_acyclic_constr_mc()
          │     └─ log_full_likelihood() + log_theta_prior()
          └─ grad_theta_log_joint()
                └─ same two likelihood/prior paths
Z latent ([D, K, 2]) encodes per-edge logits with the Innvae / Fortuin u–v factorisation (Eq. 9 in the paper).

θ is the (currently linear-SEM) weight matrix.

gumbel_soft_gmat() draws differentiable Gumbel-sigmoid edge samples; bernoulli_soft_gmat() produces element-wise probabilities.

acyclic_constr() implements the NOTEARS trace constraint h(G)=tr((I+αG)^d)-d. (Batch-aware patch included below.)

Annealing: update_dibs_hparams() rescales α and β each global step (t) with a warm-up factor so the DAG prior starts gentle and ramps up over time.

2 . Current experiment (chain X₁→X₂→X₃)
Ground truth:

makefile
Copy
Edit
X1   ~ N(0,1)
X2   = +2.0 · X1   + ε,  ε~N(0,σ²)
X3   = -1.5 · X2   + ε
Learning loop: 100 iterations of gradient ascent, step-sizes 0.005 for both Z and θ, gradient-norm clipping (11 and 19 by default).

Outcome: converges to the correct DAG within 50–70 iterations; coefficient estimates within ~10 % of ground truth.

3 . Known pain-points & open questions
🔧 Symptom	Likely cause	Mitigation ideas
Exploding θ gradients when true coefficients are large (≈10 or ≈-5).	Loss surface becomes stiff; Gaussian prior σ too small.	Either raise theta_prior_sigma or switch optimiser to Adam with per-parameter adaptivity; tune a “warm start” period for β.
Learnt graph cyclic for some initialisations.	DAG penalty zero-grad due to torch.bernoulli in gumbel_acyclic_constr_mc (non-diff).	Use soft Gumbel matrix inside h(G) (patch below).
trace dim-3 error beyond iter 80 (reported).	Batched graph input to acyclic_constr, but function assumed 2-D.	Add batch-aware trace (see patch).
Hydra config noise: many unused flags / comments.	Prototype still in flux.	Cull dead keys; move defaults into conf/config.yaml; add CI that runs pytest -q.