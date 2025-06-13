import torch
import igraph as ig
import numpy as np
import logging
from typing import Callable

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _topological_sort(g):
    # igraph's topological sort returns vertex indices in topological order
    return g.topological_sorting(mode='out')

def _permute_adjacency(adj_matrix, perm):
    """
    Permutes the rows and columns of an adjacency matrix according to a permutation.
    """
    perm_tensor = torch.LongTensor(perm)
    return adj_matrix[perm_tensor][:, perm_tensor]

def linear_functional_relationship(parents: torch.Tensor) -> torch.Tensor:
# a simple sum of parents as the functional relationship
    return torch.sum(parents, dim=1)

def generate_erdos_renyi_dag(n_nodes: int, p_edge: float, n_samples: int, functional_relationship: Callable = linear_functional_relationship):
    """
    Generates a synthetic dataset from an Erdős-Rényi DAG.

    Args:
        n_nodes: Number of nodes in the graph.
        p_edge: Probability of an edge between any two nodes.
        n_samples: Number of data points to generate.
        functional_relationship: A callable that defines the structural equation.

    Returns:
        A tuple containing:
            - The adjacency matrix of the generated DAG as a torch.Tensor.
            - The generated data as a torch.Tensor.
    """
    adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    # Nodes are implicitly ordered 0, 1, ..., n_nodes - 1
    # Edges only go from a node i to a node j where i < j
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if torch.rand(1).item() < p_edge:
                adj_matrix[i, j] = 1  # Edge from i to j
    logging.info("Generated a valid Erdős-Rényi DAG constructively.")
    data = sample_from_graph(adj_matrix, n_samples, functional_relationship=functional_relationship)
    return adj_matrix, data



def generate_scale_free_dag(n_nodes: int, m_edges: int, n_samples: int, functional_relationship: Callable = linear_functional_relationship):
    """
    Generates a synthetic dataset from a scale-free DAG (Barabasi-Albert model).

    Args:
        n_nodes: Number of nodes in the graph.
        m_edges: Number of edges to attach from a new node to existing nodes.
        n_samples: Number of data points to generate.
        functional_relationship: A callable that defines the structural equation.

    Returns:
        A tuple containing:
            - The adjacency matrix of the generated DAG as a torch.Tensor.
            - The generated data as a torch.Tensor.
    """
    while True:
        # Generate a random graph
        g = ig.Graph.Barabasi(n=n_nodes, m=m_edges, directed=True)
        
        if g.is_dag():
            logging.info("Generated a valid Scale-Free DAG.")
            topo_order = _topological_sort(g)
            adj_matrix = torch.tensor(np.array(g.get_adjacency().data), dtype=torch.float32)
            adj_matrix = _permute_adjacency(adj_matrix, topo_order)
            data = sample_from_graph(adj_matrix, n_samples, functional_relationship=functional_relationship)
            return adj_matrix, data

def sample_from_graph(adj_matrix: torch.Tensor, n_samples: int, functional_relationship: Callable = linear_functional_relationship, noise_std: float = 0.1):
    """
    Samples data from a given DAG using a specified functional relationship.

    Args:
        adj_matrix: The adjacency matrix of the DAG (torch.Tensor), must be topologically sorted.
        n_samples: Number of data points to generate.
        functional_relationship: A callable defining the structural equation for a node given its parents.
        noise_std: The standard deviation of the Gaussian noise.

    Returns:
        The generated data as a torch.Tensor of shape (n_samples, n_nodes).
    """
    n_nodes = adj_matrix.shape[0]
    data = torch.zeros(n_samples, n_nodes)
    
    for i in range(n_nodes):
        parent_indices = (adj_matrix[:, i] == 1).nonzero(as_tuple=True)[0]
        noise = torch.randn(n_samples) * noise_std
        
        if len(parent_indices) == 0:
            # Root node: value is just noise
            node_values = noise
        else:
            # Child node: value is a function of parents + noise
            parents = data[:, parent_indices]
            deterministic_part = functional_relationship(parents)
            node_values = deterministic_part + noise
            
        data[:, i] = node_values
        
    return data
