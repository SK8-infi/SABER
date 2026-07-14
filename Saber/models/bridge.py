import math
import torch
import torch.nn as nn
from typing import Tuple

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time steps, more expressive than MLP."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t * freqs.unsqueeze(0)
        embed = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.mlp(embed)

class ResBlockCFM(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.ln1(self.fc1(x))
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        return x + self.ln2(self.fc2(h))

class AttentionBlockCFM(nn.Module):
    """Self-attention block with time conditioning for CFM bridge."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, D) → (B, 1, D) for attention
        x_seq = x.unsqueeze(1)
        q_bias = self.time_proj(t_emb).unsqueeze(1)
        x_norm = self.ln(x_seq + q_bias)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return x + attn_out.squeeze(1)

class CFMBridge(nn.Module):
    def __init__(self, dim: int = 384, hidden_dim: int = 768, num_blocks: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.in_proj = nn.Linear(dim * 2, hidden_dim)

        # Interleave ResBlocks with AttentionBlocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlockCFM(hidden_dim, hidden_dim, dropout=dropout))
            if (i + 1) % 2 == 0:  # Add attention every 2 ResBlocks
                self.blocks.append(AttentionBlockCFM(hidden_dim, num_heads=4, dropout=dropout))

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
        logvar = torch.clamp(logvar, min=-10.0, max=5.0)

        return v, logvar

class CFMBridgeWrapper(nn.Module):
    def __init__(self, cfm_bridge: nn.Module, ode_steps: int = 10) -> None:
        super().__init__()
        self.cfm_bridge = cfm_bridge
        self.ode_steps = ode_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Integrate ODE dz/d_tau = v(z, tau, x) to map source → target latent
        z = x.clone()
        device = x.device
        if self.ode_steps == 1:
            tau = torch.zeros(z.shape[0], 1, device=device)
            v, _ = self.cfm_bridge(z, tau, x)
            z = z + v
        else:
            dt = 1.0 / self.ode_steps
            for step in range(self.ode_steps):
                tau = torch.ones(z.shape[0], 1, device=device) * (step * dt)
                v, _ = self.cfm_bridge(z, tau, x)
                z = z + v * dt
        return z
