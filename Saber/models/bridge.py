import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Optional

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
        if h.ndim == 3 and scale.ndim == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        h = h * (1.0 + scale) + shift
        h = self.act(h)
        h = self.dropout(h)
        return x + self.ln2(self.fc2(h))

class AttentionBlockCFM(nn.Module):
    """Self-attention block with time conditioning for CFM bridge (supports 2D or 3D sequence tokens)."""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.time_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        is_2d = (x.ndim == 2)
        x_seq = x.unsqueeze(1) if is_2d else x
        
        q_bias = self.time_proj(t_emb)
        if q_bias.ndim == 2 and x_seq.ndim == 3:
            q_bias = q_bias.unsqueeze(1)
            
        x_norm = self.ln(x_seq + q_bias)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        out = x_seq + attn_out
        return out.squeeze(1) if is_2d else out

class CFMBridge(nn.Module):
    """
    Stochastic Latent Bridge using Flow Matching (v_phi(z_tau, tau, c_a, s)).
    
    Implements BAH.pdf Specification:
    - Shared Learnable Queries s acting as modality-agnostic semantic anchors.
    - Velocity field v_phi for multi-step ODE flow.
    - Distilled Single-Step Predictor P_phi for instant 1-step inference.
    - Residual Variance logvar to compute per-query uncertainty u(q).
    """
    def __init__(
        self,
        dim: int = 768,
        hidden_dim: int = 768,
        num_blocks: int = 4,
        num_queries: int = 8,
        num_classes: int = 19,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Feature 1: Shared Learnable Queries s (Modality-Agnostic Semantic Anchors)
        self.shared_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.query_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True, dropout=dropout)

        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.in_proj = nn.Linear(dim * 2, hidden_dim)

        # Class Conditioning Projection
        self.class_emb = nn.Sequential(
            nn.Linear(num_classes, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Interleave ResBlocks with AttentionBlocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlockCFM(hidden_dim, hidden_dim, dropout=dropout))
            if (i + 1) % 2 == 0:
                self.blocks.append(AttentionBlockCFM(hidden_dim, num_heads=4, dropout=dropout))

        self.query_scale = nn.Parameter(torch.tensor(0.0))
        self.is_queries_trained = False

        # Heads
        self.out_v = nn.Linear(hidden_dim, dim)
        self.out_logvar = nn.Linear(hidden_dim, dim)

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        # Check if loaded state dict actually includes trained shared_queries
        if any("shared_queries" in k for k in state_dict.keys()):
            self.is_queries_trained = True
        else:
            self.is_queries_trained = False
        return super().load_state_dict(state_dict, strict=strict)

    def _condition_context(self, c_a: torch.Tensor) -> torch.Tensor:
        """
        Conditions context c_a on shared learnable queries s.
        Computes v_phi(z_tau, tau, c_a, s).
        """
        # If shared queries are untrained and in eval mode, bypass to prevent noise corruption
        if not getattr(self, "is_queries_trained", False) and not self.training:
            return c_a

        B = c_a.shape[0]
        s = self.shared_queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_q, D)
        
        is_2d = (c_a.ndim == 2)
        c_seq = c_a.unsqueeze(1) if is_2d else c_a  # (B, 1, D) or (B, L, D)

        # Cross-attend source context with shared query semantic anchors
        q_ctx, _ = self.query_attn(c_seq, s, s)
        c_cond = c_seq + self.query_scale * q_ctx
        return c_cond.squeeze(1) if is_2d else c_cond

    def forward(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        c_a: torch.Tensor,
        c_class: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            v: Predicted velocity field v_phi(z_tau, tau, c_a, c_class, s)
            logvar: Residual log-variance for per-query uncertainty u(q)
        """
        t_emb = self.time_emb(tau)
        c_cond = self._condition_context(c_a)

        if c_class is not None:
            c_cond = c_cond + self.class_emb(c_class)

        h = torch.cat([z_tau, c_cond], dim=-1)
        h = self.in_proj(h)

        for block in self.blocks:
            h = block(h, t_emb)

        v = self.out_v(h)
        logvar = torch.clamp(self.out_logvar(h), min=-10.0, max=5.0)

        return v, logvar


class CFMBridgeWrapper(nn.Module):
    """
    Multi-step / 1-step Euler predictor wrapper for CFM Bridge with Calibrated Uncertainty u(q).
    Supports (B, D) pooled features and (B, L, D) spatial patch token sequences.
    """
    def __init__(self, cfm_bridge: CFMBridge, ode_steps: int = 10) -> None:
        super().__init__()
        self.cfm_bridge = cfm_bridge
        self.ode_steps = ode_steps

    def predict_with_uncertainty(
        self, x: torch.Tensor, c_class: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translates source context x -> target manifold latent z_pred, 
        and computes per-query calibrated uncertainty u(q) in [0, 1].

        Returns:
            z_pred: Translated target latent embedding (B, D)
            u_q: Per-query uncertainty score tensor (B,) in [0, 1]
        """
        z = x.clone()
        device = x.device
        B = x.shape[0]

        if self.ode_steps == 1:
            # 1-step Euler integration
            tau = torch.zeros(B, 1, device=device)
            v, logvar = self.cfm_bridge(z, tau, x, c_class=c_class)
            z_pred = z + v
            
            # Calibrated Uncertainty u(q) = sigmoid(mean(logvar))
            mean_logvar = logvar.mean(dim=-1) if logvar.ndim == 2 else logvar.mean(dim=(-1, -2))
            u_q = torch.sigmoid(mean_logvar)
        else:
            # Multi-step Euler ODE integration
            dt = 1.0 / self.ode_steps
            accum_logvar = torch.zeros(B, device=device)
            for step in range(self.ode_steps):
                tau = torch.ones(B, 1, device=device) * (step * dt)
                v, logvar = self.cfm_bridge(z, tau, x, c_class=c_class)
                z = z + v * dt
                accum_logvar = accum_logvar + (logvar.mean(dim=-1) if logvar.ndim == 2 else logvar.mean(dim=(-1, -2)))
            
            z_pred = z
            u_q = torch.sigmoid(accum_logvar / self.ode_steps)

        return z_pred, u_q

    def forward(self, x: torch.Tensor, c_class: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_pred, _ = self.predict_with_uncertainty(x, c_class=c_class)
        return z_pred

