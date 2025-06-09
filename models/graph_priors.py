import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
import igraph as ig

def generate_erdos_renyi_dag(
    d: int, 
    p_edge: float, 
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate a DAG using Erdős-Rényi model with igraph."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Generate random directed graph
    g = ig.Graph.Erdos_Renyi(n=d, p=p_edge, directed=True)
    
    # Debug: Check if graph has edges
    print(f"DEBUG: Generated ER graph with {g.ecount()} edges out of {d*(d-1)} possible")
    
    # Convert to DAG by removing back edges (topological ordering)
    if not g.is_dag():
        # Get all edges and remove those that create cycles
        edges_to_remove = []
        for edge in g.es:
            source, target = edge.tuple
            if source >= target:  # Simple ordering: remove backward edges
                edges_to_remove.append(edge.index)
        
        print(f"DEBUG: Removing {len(edges_to_remove)} backward edges to make DAG")
        g.delete_edges(edges_to_remove)
    
    print(f"DEBUG: Final DAG has {g.ecount()} edges")
    
    # Convert to adjacency matrix
    adj_matrix = np.array(g.get_adjacency().data)
    return torch.tensor(adj_matrix, dtype=torch.float32)

def generate_scale_free_dag(
    d: int, 
    m: int = 2, 
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate a DAG using scale-free (Barabási-Albert) model with igraph."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Ensure m is not too large
    m = min(m, d-1)
    
    # Generate scale-free graph (undirected first)
    g = ig.Graph.Barabasi(n=d, m=m, directed=False)
    
    print(f"DEBUG: Generated Barabasi graph with {g.ecount()} edges")
    
    # Convert to directed by creating directed edges from undirected ones
    # For each undirected edge (i,j), create directed edge from min(i,j) to max(i,j)
    directed_edges = []
    for edge in g.es:
        source, target = edge.tuple
        if source != target:  # Avoid self-loops
            # Always direct from lower to higher index to ensure DAG property
            directed_edges.append((min(source, target), max(source, target)))
    
    # Create new directed graph
    g_directed = ig.Graph(n=d, edges=directed_edges, directed=True)
    
    print(f"DEBUG: Final scale-free DAG has {g_directed.ecount()} edges")
    
    # Convert to adjacency matrix
    adj_matrix = np.array(g_directed.get_adjacency().data)
    return torch.tensor(adj_matrix, dtype=torch.float32)

def generate_chain_dag(d: int) -> torch.Tensor:
    """Generate a simple chain DAG: 0→1→2→...→(d-1)."""
    G = torch.zeros(d, d)
    for i in range(d - 1):
        G[i, i + 1] = 1.0
    return G

# Also fix the weight sampling to ensure reasonable weights
def sample_edge_weights(
    G: torch.Tensor, 
    weight_range: Tuple[float, float] = (-2.0, 2.0),
    seed: Optional[int] = None
) -> torch.Tensor:
    """Sample edge weights for a given graph structure."""
    if seed is not None:
        torch.manual_seed(seed)
    
    d = G.shape[0]
    weights = torch.zeros_like(G)
    
    # Sample weights only for existing edges
    edges = torch.nonzero(G, as_tuple=False)
    print(f"DEBUG: Sampling weights for {len(edges)} edges")
    
    for edge in edges:
        i, j = edge[0].item(), edge[1].item()
        # Sample weight in range, ensuring it's not too small
        weight = torch.rand(1).item() * (weight_range[1] - weight_range[0]) + weight_range[0]
        
        # Ensure minimum magnitude
        if abs(weight) < 0.5:
            weight = 0.5 if weight >= 0 else -0.5
            
        weights[i, j] = weight
    
    print(f"DEBUG: Sampled weights range: {weights[weights != 0].min().item():.3f} to {weights[weights != 0].max().item():.3f}")
    
    return weights

def generate_linear_sem_data(
    G: torch.Tensor,
    Theta: torch.Tensor, 
    n_samples: int,
    noise_std: float = 0.1,
    seed: Optional[int] = None
) -> torch.Tensor:
    """Generate data from linear SEM: X = Theta * X + noise."""
    if seed is not None:
        torch.manual_seed(seed)
    
    d = G.shape[0]
    X = torch.zeros(n_samples, d)
    
    # Topological order (assuming G is a DAG)
    # Simple ordering: process nodes 0, 1, 2, ... (works for our generated DAGs)
    for j in range(d):
        # X_j = sum_i Theta_ij * X_i + noise_j
        parents = torch.nonzero(G[:, j], as_tuple=False).flatten()
        
        if len(parents) > 0:
            parent_contribution = torch.sum(
                X[:, parents] * Theta[parents, j].unsqueeze(0), 
                dim=1
            )
        else:
            parent_contribution = torch.zeros(n_samples)
        
        noise = torch.randn(n_samples) * noise_std
        X[:, j] = parent_contribution + noise
    
    return X

def generate_hardcoded_chain_data(
    num_samples: int, 
    obs_noise_std: float, 
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate the exact hardcoded 3-node chain: X1 -> X2 -> X3
    This is NOT a prior - it's a specific test case with known weights.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # HARDCODED structure
    G_true = torch.zeros(3, 3, dtype=torch.float32)
    G_true[0, 1] = 1.0  # X1 -> X2
    G_true[1, 2] = 1.0  # X2 -> X3

    # HARDCODED weights (same as original working version)
    Theta_true = torch.zeros(3, 3, dtype=torch.float32)
    Theta_true[0, 1] = 2.0   # X2 = 2.0 * X1 + noise
    Theta_true[1, 2] = -1.5  # X3 = -1.5 * X2 + noise

    # Generate data (same as original)
    X_data = torch.zeros(num_samples, 3)
    X_data[:, 0] = torch.randn(num_samples)  # X1 ~ N(0,1)
    
    noise_x2 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 1] = Theta_true[0, 1] * X_data[:, 0] + noise_x2
    
    noise_x3 = torch.randn(num_samples) * obs_noise_std
    X_data[:, 2] = Theta_true[1, 2] * X_data[:, 1] + noise_x3
    
    return {
        'x': X_data, 
        'G_true': G_true, 
        'Theta_true': Theta_true, 
        'y': None
    }

def generate_ground_truth_data_with_graph_prior(
    graph_type: str,
    d: int,
    num_samples: int,
    obs_noise_std: float = 0.1,
    graph_params: Optional[Dict] = None,
    weight_range: Tuple[float, float] = (-2.0, 2.0),
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """Generate synthetic data with different approaches."""
    
    if graph_params is None:
        graph_params = {}
    
    # SPECIAL CASE: Hardcoded test chain (not a real prior)
    if graph_type == "chain" and d == 3:
        return generate_hardcoded_chain_data(num_samples, obs_noise_std, seed)
    
    # REAL PRIORS: Generate random graphs with sampled weights
    if graph_type == "erdos_renyi":
        p_edge = graph_params.get("p_edge", 0.3)
        G = generate_erdos_renyi_dag(d, p_edge, seed)
    elif graph_type == "scale_free":
        m = graph_params.get("m", 2)
        G = generate_scale_free_dag(d, m, seed)
    elif graph_type == "chain":
        # For chain with d != 3, generate random chain
        G = generate_chain_dag(d)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Sample random weights for the generated structure
    Theta = sample_edge_weights(G, weight_range, seed)
    
    # Generate data
    X = generate_linear_sem_data(G, Theta, num_samples, obs_noise_std, seed)
    
    return {
        'x': X,
        'y': None,
        'G_true': G,
        'Theta_true': Theta
    }

# Backward compatibility: keep the original function name and signature
def generate_ground_truth_data_x1_x2_x3(
    num_samples: int, 
    obs_noise_std: float, 
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Original function for backward compatibility.
    Generates the same 3-node chain as before.
    """
    return generate_ground_truth_data_with_graph_prior(
        graph_type="chain",
        d=3,
        num_samples=num_samples,
        obs_noise_std=obs_noise_std,
        weight_range=(-2.0, 2.0),
        seed=seed
    )