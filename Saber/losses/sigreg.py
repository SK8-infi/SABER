import torch

def sigreg_strong_loss(x: torch.Tensor, sketch_dim: int = 64, num_points: int = 17) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) - Strong formulation.
    Forces ECF(x) ~ ECF(Gaussian). Matches ALL Moments (Maximum Entropy Cloud).
    Based on LeJEPA Algorithm 1.
    
    Args:
        x: Input embedding tensor of shape (N, C).
        sketch_dim: Number of random slice projection directions (default 64).
        num_points: Number of integration frequency points (default 17).
        
    Returns:
        Scalar loss tensor.
    """
    N, C = x.size()
    if N <= 1:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)

    # Cast to float32 to prevent precision loss or NaN during autocast float16 training
    x_32 = x.float()
    
    # 1. Random Projection Matrix (Cramér-Wold slice directions)
    A = torch.randn(C, sketch_dim, device=x_32.device, dtype=torch.float32)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)
    
    # 2. Project embeddings
    proj = torch.matmul(x_32, A)  # Shape: (N, sketch_dim)
    
    # 3. Integration Points t_k uniformly in [0.0, 3.0]
    t = torch.linspace(0.0, 3.0, num_points, device=x_32.device, dtype=torch.float32)
    
    # 4. Target Gaussian CF: phi(t_k) = exp(-t_k^2 / 2)
    phi = torch.exp(-0.5 * t**2)
    
    # 5. Compute ECF projections: args has shape (N, sketch_dim, num_points)
    args = proj.unsqueeze(2) * t.view(1, 1, -1)
    
    # 6. ECF components averaged over batch dimension (dim=0)
    ecf_cos = torch.cos(args).mean(dim=0)  # Shape: (sketch_dim, num_points)
    ecf_sin = torch.sin(args).mean(dim=0)  # Shape: (sketch_dim, num_points)
    
    # 7. Weighted L2 Distance
    diff_cos = ecf_cos - phi.unsqueeze(0)
    diff_sin = ecf_sin
    diff_sq = diff_cos.pow(2) + diff_sin.pow(2)
    
    # err shape: (sketch_dim, num_points)
    err = diff_sq * phi.unsqueeze(0)
    
    # Integrate over the t dimension (dim=-1)
    # trapz/trapezoid returns shape (sketch_dim,)
    if hasattr(torch, "trapezoid"):
        loss = torch.trapezoid(err, t, dim=-1)
    else:
        loss = torch.trapz(err, t, dim=-1)
        
    return loss.mean()
