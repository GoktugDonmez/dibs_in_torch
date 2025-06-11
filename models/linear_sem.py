import torch

__all__ = ["linear_predict"]

def linear_predict(x: torch.Tensor, theta: torch.Tensor, g_soft: torch.Tensor) -> torch.Tensor:
    """Compute predictions for a linear SEM."""
    effective_w = theta * g_soft
    return torch.matmul(x, effective_w)
