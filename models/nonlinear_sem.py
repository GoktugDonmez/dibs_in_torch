import torch
import torch.nn as nn

class NodeMLP(nn.Module):
    def __init__(self, d: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NonlinearSEM(nn.Module):
    """Simple MLP-based SEM where each node has its own MLP."""

    def __init__(self, d: int, hidden_dim: int = 16):
        super().__init__()
        self.d = d
        self.mlps = nn.ModuleList([NodeMLP(d, hidden_dim) for _ in range(d)])

    def forward(self, x: torch.Tensor, g_soft: torch.Tensor) -> torch.Tensor:
        outputs = []
        for j in range(self.d):
            mask = g_soft[:, j]
            masked_x = x * mask
            out_j = self.mlps[j](masked_x).squeeze(1)
            outputs.append(out_j)
        return torch.stack(outputs, dim=1)
