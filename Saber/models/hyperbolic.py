"""
SABER Hyperbolic Geometry Module (geoopt Poincaré Ball Integration)
===================================================================
Provides Poincaré Ball manifold operations, exponential maps, Mobius operations,
and Hyperbolic Distance computation for hierarchical land-cover representation learning.
Supports both native PyTorch implementation and official geoopt package integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geoopt
    HAS_GEOOPT = True
except ImportError:
    geoopt = None
    HAS_GEOOPT = False

def artanh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x_32 = x.float()
    x_clamped = torch.clamp(x_32, -1.0 + eps, 1.0 - eps)
    return 0.5 * torch.log((1.0 + x_clamped) / (1.0 - x_clamped) + eps)

def arcosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x_32 = x.float()
    x_clamped = torch.clamp(x_32, min=1.0 + eps)
    return torch.log(x_clamped + torch.sqrt(x_clamped.pow(2) - 1.0 + eps))


class NativePoincareBall:
    """
    Native PyTorch implementation of Poincaré Ball Manifold B^d (c = 1.0).
    Exposes identical interface to geoopt.PoincareBall for zero-dependency execution.
    """
    def __init__(self, c: float = 1.0, eps: float = 1e-5):
        self.c = c
        self.sqrt_c = math.sqrt(c)
        self.eps = eps

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Projects points into interior of Poincaré Ball ||x|| <= (1 - eps) / sqrt(c)."""
        x_32 = x.float()
        norm = torch.norm(x_32, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = (1.0 - self.eps) / self.sqrt_c
        cond = norm > max_norm
        projected = x_32 / norm * max_norm
        res = torch.where(cond, projected, x_32)
        return res.to(x.dtype)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space at origin T_0 B^d to Poincaré Ball B^d."""
        v_32 = v.float()
        v_norm = torch.norm(v_32, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gamma = torch.tanh(self.sqrt_c * v_norm) * (v_32 / (self.sqrt_c * v_norm))
        return self.proj(gamma).to(v.dtype)

    def logmap0(self, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from Poincaré Ball B^d to tangent space at origin T_0 B^d."""
        y_proj = self.proj(y).float()
        y_norm = torch.norm(y_proj, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        res = artanh(self.sqrt_c * y_norm) * (y_proj / (self.sqrt_c * y_norm))
        return res.to(y.dtype)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition x (+) y in Poincaré Ball."""
        x_32 = x.float()
        y_32 = y.float()
        x2 = torch.sum(x_32 * x_32, dim=-1, keepdim=True)
        y2 = torch.sum(y_32 * y_32, dim=-1, keepdim=True)
        xy = torch.sum(x_32 * y_32, dim=-1, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x_32 + (1 - self.c * x2) * y_32
        denom = 1 + 2 * self.c * xy + (self.c ** 2) * x2 * y2
        res = self.proj(num / denom.clamp(min=1e-8))
        return res.to(x.dtype)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance d_H(x, y) in Poincaré Ball."""
        diff = self.mobius_add(-x, y).float()
        diff_norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        res = (2.0 / self.sqrt_c) * artanh(self.sqrt_c * diff_norm)
        return res.squeeze(-1)

    def dist_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes pairwise Hyperbolic Distance matrix d_H(x_i, y_j)."""
        x_proj = self.proj(x).float()
        y_proj = self.proj(y).float()
        x2 = torch.sum(x_proj ** 2, dim=-1, keepdim=True)    # (N, 1)
        y2 = torch.sum(y_proj ** 2, dim=-1, keepdim=True).T  # (1, M)
        sqdist = torch.cdist(x_proj, y_proj, p=2).pow(2)     # (N, M)
        
        num = 2 * sqdist
        denom = (1.0 - self.c * x2) * (1.0 - self.c * y2)
        arg = 1.0 + num / denom.clamp(min=1e-7)
        return arcosh(arg) / self.sqrt_c


def get_poincare_ball(c: float = 1.0):
    """Returns geoopt.PoincareBall if package installed, else NativePoincareBall."""
    if HAS_GEOOPT:
        return geoopt.PoincareBall(c=c)
    return NativePoincareBall(c=c)


class PoincareProjectionHead(nn.Module):
    """
    Hyperbolic Projection Head projecting linear features onto the Poincaré Ball.
    Linear -> LayerNorm -> GELU -> Linear -> expmap0 -> PoincareBall Embedding
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, c: float = 1.0):
        super().__init__()
        self.manifold = get_poincare_ball(c=c)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.ln(self.fc1(x)))
        v = self.fc2(h)
        # Project tangent vector v at origin into Poincaré Ball B^d
        if HAS_GEOOPT and hasattr(self.manifold, "expmap0"):
            z_hyp = self.manifold.expmap0(v)
            return self.manifold.projx(z_hyp)
        else:
            return self.manifold.expmap0(v)


def poincare_loss(z1: torch.Tensor, z2: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Computes mean Hyperbolic Poincaré distance loss d_H(z1, z2)^2."""
    ball = get_poincare_ball(c=c)
    if HAS_GEOOPT and hasattr(ball, "dist"):
        d = ball.dist(z1, z2)
    else:
        d = ball.dist(z1, z2)
    return torch.mean(d.pow(2))
