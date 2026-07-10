import torch
import torch.nn as nn
from typing import Tuple

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        return self.mlp(t)

class ResBlockCFM(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.ln1(self.fc1(x))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        return x + self.fc2(h)

class CFMBridge(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 512, num_blocks: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.time_emb = TimeEmbedding(hidden_dim)
        self.in_proj = nn.Linear(dim * 2, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResBlockCFM(hidden_dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        self.out_v = nn.Linear(hidden_dim, dim)
        self.out_logvar = nn.Linear(hidden_dim, dim)

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.time_emb(tau)
        h = torch.cat([z_tau, c], dim=-1)
        h = self.in_proj(h)
        
        for block in self.blocks:
            h = block(h, t_emb)
            
        v = self.out_v(h)
        logvar = self.out_logvar(h)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)
        
        return v, logvar
