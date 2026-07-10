import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from Saber.models.bridge import CFMBridge
from Saber.losses.bridge_loss import CFMLoss
from Saber.trainer.metrics import compute_retrieval_metrics
from Saber.utils.seed import set_seed

def integrate_ode(model, z_s1, steps=1, device="cpu"):
    """
    Integrates the ODE dz/d_tau = v(z, tau, z_s1) using Euler integration.
    """
    model.eval()
    with torch.no_grad():
        z = z_s1.clone().to(device)
        if steps == 1:
            # Fast 1-step prediction
            tau = torch.zeros(z.shape[0], 1, device=device)
            v, _ = model(z, tau, z_s1.to(device))
            z = z + v
        else:
            dt = 1.0 / steps
            for step in range(steps):
                tau = torch.ones(z.shape[0], 1, device=device) * (step * dt)
                v, _ = model(z, tau, z_s1.to(device))
                z = z + v * dt
        return z

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFM Latent Bridge on extracted features")
    parser.add_argument("--features_dir", type=str, default="Saber_bridge/extracted", help="Directory with extracted features")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ode_steps", type=int, default=5, help="ODE solver integration steps for evaluation")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load extracted features
    try:
        s1_feats = np.load(os.path.join(args.features_dir, "s1_feats.npy"))
        s2_feats = np.load(os.path.join(args.features_dir, "s2_feats.npy"))
        labels = np.load(os.path.join(args.features_dir, "labels.npy"))
        print(f"Loaded features from '{args.features_dir}'")
        print(f"s1_feats: {s1_feats.shape}, s2_feats: {s2_feats.shape}, labels: {labels.shape}")
    except Exception as e:
        print(f"Failed to load features: {e}")
        sys.exit(1)

    num_samples = s1_feats.shape[0]
    
    # 80/20 query/gallery partition mirroring Evaluator
    rng = np.random.RandomState(42)
    shuffled_indices = rng.permutation(num_samples)
    query_size = max(1, num_samples // 5)
    query_indices = np.sort(shuffled_indices[:query_size])
    gallery_indices = np.sort(shuffled_indices[query_size:])

    # Train uses gallery partition (80%)
    train_s1 = torch.tensor(s1_feats[gallery_indices], dtype=torch.float32)
    train_s2 = torch.tensor(s2_feats[gallery_indices], dtype=torch.float32)

    # Val uses query partition (20%)
    val_s1 = torch.tensor(s1_feats[query_indices], dtype=torch.float32)
    val_s2 = torch.tensor(s2_feats[query_indices], dtype=torch.float32)
    val_lbl = labels[query_indices]

    train_dataset = TensorDataset(train_s1, train_s2)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model, loss and optimization
    model = CFMBridge(dim=384, hidden_dim=512, num_blocks=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    loss_fn = CFMLoss()

    best_f1 = 0.0
    best_epoch = 0

    print("Starting CFM bridge training loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch_s1, batch_s2 in train_loader:
            batch_s1, batch_s2 = batch_s1.to(device), batch_s2.to(device)
            
            optimizer.zero_grad()
            
            # Sample time steps tau ~ U(0, 1)
            tau = torch.rand(batch_s1.size(0), 1, device=device)
            
            # Interpolate latents z_tau
            z_tau = (1.0 - tau) * batch_s1 + tau * batch_s2
            
            # Predict velocity and log-variance
            pred_v, logvar = model(z_tau, tau, batch_s1)
            
            # Loss computation
            loss = loss_fn(pred_v, logvar, batch_s1, batch_s2)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_s1.size(0)

        train_loss /= len(train_dataset)
        scheduler.step()

        # Evaluation (both 1-step and multi-step ODE solvers)
        model.eval()
        with torch.no_grad():
            # 1-step Euler prediction
            pred_val_s2_1step = integrate_ode(model, val_s1, steps=1, device=device)
            pred_val_s2_1step_norm = F.normalize(pred_val_s2_1step, dim=-1).cpu().numpy()
            
            # N-step Euler prediction
            pred_val_s2_nstep = integrate_ode(model, val_s1, steps=args.ode_steps, device=device)
            pred_val_s2_nstep_norm = F.normalize(pred_val_s2_nstep, dim=-1).cpu().numpy()
            
            val_s2_norm = F.normalize(val_s2, dim=-1).numpy()
            
            # Compute evaluation retrieval metrics (1-step vs N-step)
            metrics_1step = compute_retrieval_metrics(
                query_embeds=pred_val_s2_1step_norm,
                gallery_embeds=val_s2_norm,
                query_labels=val_lbl,
                gallery_labels=val_lbl,
                top_k=5,
                is_multilabel=True
            )
            
            metrics_nstep = compute_retrieval_metrics(
                query_embeds=pred_val_s2_nstep_norm,
                gallery_embeds=val_s2_norm,
                query_labels=val_lbl,
                gallery_labels=val_lbl,
                top_k=5,
                is_multilabel=True
            )

        f1_1step = metrics_1step["f1@5"]
        f1_nstep = metrics_nstep["f1@5"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {train_loss:.4f} | "
              f"1-Step F1@5: {f1_1step:.4f} | "
              f"{args.ode_steps}-Step F1@5: {f1_nstep:.4f}")

        # Choose the N-step configuration for checkpoints
        if f1_nstep > best_f1:
            best_f1 = f1_nstep
            best_epoch = epoch
            os.makedirs("Saber_bridge/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "Saber_bridge/checkpoints/bridge_best.pth")

    print(f"\nTraining finished! Best F1@5: {best_f1:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()
