import torch
import numpy as np
from typing import Dict, Tuple, Optional
import igraph as ig

__all__ = [
    "generate_erdos_renyi_dag",
    "generate_scale_free_dag",
    "generate_chain_dag",
    "sample_edge_weights",
    "generate_linear_sem_data",
    "generate_hardcoded_chain_data",
    "generate_ground_truth_data_with_graph_prior",
    "generate_ground_truth_data_x1_x2_x3",
]


def generate_erdos_renyi_dag(d: int, p_edge: float, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a DAG using an Erd\u0151s-R\u00e9nyi prior."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    g = ig.Graph.Erdos_Renyi(n=d, p=p_edge, directed=True)

    if not g.is_dag():
        edges_to_remove = []
        for edge in g.es:
            source, target = edge.tuple
            if source >= target:
                edges_to_remove.append(edge.index)
        g.delete_edges(edges_to_remove)

    adj_matrix = np.array(g.get_adjacency().data)
    return torch.tensor(adj_matrix, dtype=torch.float32)


def generate_scale_free_dag(d: int, m: int = 2, seed: Optional[int] = None) -> torch.Tensor:
    """Generate a DAG from a scale-free prior."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    m = min(m, d - 1)
    g = ig.Graph.Barabasi(n=d, m=m, directed=False)

    directed_edges = []
    for edge in g.es:
        s, t = edge.tuple
        if s != t:
            directed_edges.append((min(s, t), max(s, t)))
    g_directed = ig.Graph(n=d, edges=directed_edges, directed=True)

    adj_matrix = np.array(g_directed.get_adjacency().data)
    return torch.tensor(adj_matrix, dtype=torch.float32)


def generate_chain_dag(d: int) -> torch.Tensor:
    """Generate a simple chain DAG 0\u21921\u2192...\u2192(d-1)."""
    G = torch.zeros(d, d)
    for i in range(d - 1):
        G[i, i + 1] = 1.0
    return G


def sample_edge_weights(G: torch.Tensor, weight_range: Tuple[float, float] = (-2.0, 2.0), seed: Optional[int] = None) -> torch.Tensor:
    """Sample edge weights for a given graph structure."""
    if seed is not None:
        torch.manual_seed(seed)

    weights = torch.zeros_like(G)
    edges = torch.nonzero(G, as_tuple=False)
    for edge in edges:
        i, j = edge[0].item(), edge[1].item()
        weight = torch.rand(1).item() * (weight_range[1] - weight_range[0]) + weight_range[0]
        if abs(weight) < 0.5:
            weight = 0.5 if weight >= 0 else -0.5
        weights[i, j] = weight
    return weights


def generate_linear_sem_data(G: torch.Tensor, Theta: torch.Tensor, n_samples: int, noise_std: float = 0.1, seed: Optional[int] = None) -> torch.Tensor:
    """Generate data from a linear SEM: X = Theta * X + noise."""
    if seed is not None:
        torch.manual_seed(seed)

    d = G.shape[0]
    X = torch.zeros(n_samples, d)
    for j in range(d):
        parents = torch.nonzero(G[:, j], as_tuple=False).flatten()
        if len(parents) > 0:
            parent_contrib = torch.sum(X[:, parents] * Theta[parents, j].unsqueeze(0), dim=1)
        else:
            parent_contrib = torch.zeros(n_samples)
        noise = torch.randn(n_samples) * noise_std
        X[:, j] = parent_contrib + noise
    return X


def generate_hardcoded_chain_data(num_samples: int, obs_noise_std: float, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Generate the fixed three node chain used in tests."""
    if seed is not None:
        torch.manual_seed(seed)

    G_true = torch.zeros(3, 3, dtype=torch.float32)
    G_true[0, 1] = 1.0
    G_true[1, 2] = 1.0

    Theta_true = torch.zeros(3, 3, dtype=torch.float32)
    Theta_true[0, 1] = 2.0
    Theta_true[1, 2] = -1.5

    X_data = torch.zeros(num_samples, 3)
    X_data[:, 0] = torch.randn(num_samples)
    X_data[:, 1] = Theta_true[0, 1] * X_data[:, 0] + torch.randn(num_samples) * obs_noise_std
    X_data[:, 2] = Theta_true[1, 2] * X_data[:, 1] + torch.randn(num_samples) * obs_noise_std

    return {"x": X_data, "G_true": G_true, "Theta_true": Theta_true, "y": None}


def generate_ground_truth_data_with_graph_prior(
    graph_type: str,
    d: int,
    num_samples: int,
    obs_noise_std: float = 0.1,
    graph_params: Optional[Dict] = None,
    weight_range: Tuple[float, float] = (-2.0, 2.0),
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic data for a given graph prior."""
    if graph_params is None:
        graph_params = {}

    if graph_type == "erdos_renyi":
        p_edge = graph_params.get("p_edge", 0.3)
        G = generate_erdos_renyi_dag(d, p_edge, seed)
    elif graph_type == "scale_free":
        m = graph_params.get("m", 2)
        G = generate_scale_free_dag(d, m, seed)
    elif graph_type == "chain":
        if d == 3:
            return generate_hardcoded_chain_data(num_samples, obs_noise_std, seed)
        G = generate_chain_dag(d)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    Theta = sample_edge_weights(G, weight_range, seed)
    X = generate_linear_sem_data(G, Theta, num_samples, obs_noise_std, seed)
    return {"x": X, "y": None, "G_true": G, "Theta_true": Theta}


def generate_ground_truth_data_x1_x2_x3(num_samples: int, obs_noise_std: float, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Backward compatible helper for the old three node chain."""
    return generate_ground_truth_data_with_graph_prior(
        graph_type="chain",
        d=3,
        num_samples=num_samples,
        obs_noise_std=obs_noise_std,
        weight_range=(-2.0, 2.0),
        seed=seed,
    )
