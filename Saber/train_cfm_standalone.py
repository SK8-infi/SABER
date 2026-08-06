import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Saber.utils.config import load_config
from Saber.datasets.ben14k import BEN14KDataset
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge
from Saber.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
logger = logging.getLogger(__name__)

def train_cfm_standalone(
    config_path: str = "Saber/configs/config.yaml",
    checkpoint_path: str = "checkpoints_v10/saber_unified_clean.pth",
    data_dir: str = "Datasets/benv1_14k",
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 0.0005,
    save_dir: str = "checkpoints_v10"
):
    """
    Standalone Dedicated Training Engine for CFM Latent Bridge.
    
    1. Loads pre-trained Master Encoder (DOFA + LoRA + Projection Head).
    2. Freezes encoder completely.
    3. Trains CFM Bridge using Enhanced Multi-Objective Loss:
       - Velocity MSE Loss (||v_pred - v_target||^2)
       - Velocity Cosine Loss (1 - cos(v_pred, v_target))
       - End-Point Target Alignment Loss (1 - cos(z1 + v_pred, z2))
    4. Evaluates Cross-Modal Retrieval (S1 Query -> S2 Gallery) after every epoch.
    5. Saves best bridge checkpoint based on Cross-Modal mAP@5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info(" 🚀 SABER STANDALONE CFM LATENT BRIDGE OPTIMIZATION ENGINE")
    logger.info("=" * 80)
    logger.info(f" Device: {device} | Batch Size: {batch_size} | Epochs: {epochs} | LR: {lr}")
    logger.info(f" Encoder Checkpoint Target: '{checkpoint_path}'")
    logger.info(f" Save Directory: '{save_dir}'")

    os.makedirs(save_dir, exist_ok=True)
    config = load_config(config_path)

    # 1. Load Pre-trained Master Model
    logger.info("Instantiating SABER Master Model...")
    model = SABER(config=config, in_channels=14).to(device)

    if os.path.exists(checkpoint_path):
        logger.info(f"Loading pre-trained Master Encoder weights from '{checkpoint_path}'...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Remove any existing bridge weights to train fresh bridge
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.")}
        model.load_state_dict(state_dict, strict=False)
        logger.info("Successfully loaded master encoder weights (strict=False).")
    else:
        logger.warning(f"⚠️ Checkpoint file '{checkpoint_path}' not found! Will train bridge on current weights.")

    # Freeze Master Encoder
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 2. Instantiate Fresh CFM Bridge
    bridge_net = CFMBridge(
        dim=768,
        hidden_dim=768,
        num_blocks=4,
        num_queries=8,
        dropout=0.1
    ).to(device)

    # Mark queries as active for training
    bridge_net.is_queries_trained = True
    # Initialize query_scale to a small positive value (0.1) to allow cross-attention gradients immediately
    with torch.no_grad():
        bridge_net.query_scale.copy_(torch.tensor(0.1, device=device))

    optimizer = torch.optim.AdamW(bridge_net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Attach bridge to model for evaluator compatibility
    model.bridge.cfm_bridge = bridge_net

    # 3. Load Datasets
    logger.info(f"Loading BEN-14K dataset from '{data_dir}'...")
    train_dataset = BEN14KDataset(root_dir=data_dir, split="train", synthetic=False)
    test_dataset = BEN14KDataset(root_dir=data_dir, split="test", synthetic=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    evaluator = Evaluator(model=model, test_dataset=test_dataset, config=config, device=device)

    best_map5 = 0.0
    best_bridge_path = os.path.join(save_dir, "bridge_unified.pth")
    best_ben_path = os.path.join(save_dir, "bridge_best_ben14k.pth")

    logger.info("=" * 80)
    logger.info(" STARTING CFM LATENT BRIDGE TRAINING & REAL-TIME EVALUATION")
    logger.info("=" * 80)

    for epoch in range(1, epochs + 1):
        bridge_net.train()
        total_loss = 0.0
        total_mse = 0.0
        total_cos = 0.0
        total_end = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"CFM Epoch [{epoch}/{epochs}]", leave=True, dynamic_ncols=True)
        for batch in pbar:
            images = batch.get("image1", batch.get("image")).to(device, non_blocking=True)
            if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

            x_s1 = images[:, :2, :, :]
            x_s2 = images[:, 2:, :, :]

            # Extract frozen 768-D embeddings
            with torch.no_grad():
                feats_s1 = model.backbone(x_s1, [5.405, 5.405])
                z1 = F.normalize(model.projection_head(feats_s1), p=2, dim=-1)

                feats_s2 = model.backbone(x_s2, [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190])
                z2 = F.normalize(model.projection_head(feats_s2), p=2, dim=-1)

            optimizer.zero_grad()

            # Flow Matching Interpolation: z_tau = (1 - tau) * z1 + tau * z2
            B = z1.shape[0]
            tau = torch.rand(B, 1, device=device)
            z_tau = (1.0 - tau) * z1 + tau * z2
            v_target = z2 - z1

            # Predict velocity field
            v_pred, _ = bridge_net(z_tau, tau, z1)

            # 1. Velocity MSE Loss
            loss_mse = F.mse_loss(v_pred, v_target)

            # 2. Velocity Cosine Directional Loss
            cos_sim_v = F.cosine_similarity(v_pred, v_target, dim=-1)
            loss_cos = (1.0 - cos_sim_v).mean()

            # 3. Integrated Target Reconstruction Loss (z_hat = z1 + v_pred)
            z2_pred = F.normalize(z1 + v_pred, p=2, dim=-1)
            loss_end = (1.0 - F.cosine_similarity(z2_pred, z2, dim=-1)).mean()

            # Combined Loss
            loss_bridge = loss_mse + 0.5 * loss_cos + 0.5 * loss_end

            loss_bridge.backward()
            torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_bridge.item()
            total_mse += loss_mse.item()
            total_cos += loss_cos.item()
            total_end += loss_end.item()

            pbar.set_postfix({
                "loss": f"{loss_bridge.item():.4f}",
                "mse": f"{loss_mse.item():.4f}",
                "cos": f"{loss_cos.item():.4f}",
                "q_scale": f"{bridge_net.query_scale.item():.3f}"
            })

        scheduler.step()
        elapsed = time.time() - start_time
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_cos = total_cos / num_batches
        avg_end = total_end / num_batches

        logger.info(
            f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
            f"Bridge Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | Cos: {avg_cos:.4f} | EndPt: {avg_end:.4f} | "
            f"QueryScale: {bridge_net.query_scale.item():.3f}"
        )

        # 4. Perform Real-Time Cross-Modal Evaluation (S1 -> S2)
        logger.info(f"📊 Evaluating Epoch [{epoch}/{epochs}] Cross-Modal Retrieval (S1 Query -> S2 Gallery)...")
        bridge_net.eval()
        metrics = evaluator.evaluate_cross_modal_retrieval(query_modality="s1", gallery_modality="s2")
        map5 = metrics.get("MAP@5", 0.0)
        prec5 = metrics.get("PRECISION@5", 0.0)
        rec5 = metrics.get("RECALL@5", 0.0)

        logger.info(f"🎯 Epoch [{epoch}/{epochs}] Results -> mAP@5: {map5:.4f} | Precision@5: {prec5:.4f} | Recall@5: {rec5:.4f}")

        # Save checkpoint payload
        b_ckpt_data = {
            "epoch": epoch,
            "bridge_state_dict": bridge_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "map5": map5,
            "precision5": prec5,
            "recall5": rec5
        }

        # Save latest checkpoint
        torch.save(b_ckpt_data, best_bridge_path)

        if map5 > best_map5:
            best_map5 = map5
            torch.save(b_ckpt_data, best_ben_path)
            logger.info(f"🏆 NEW BEST Cross-Modal mAP@5: {best_map5:.4f}! Saved to '{best_ben_path}'")

    print("=" * 80)
    print(" 🎉 CFM LATENT BRIDGE TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best Cross-Modal mAP@5 : {best_map5:.4f}")
    print(f" Master Bridge Checkpoint : '{best_bridge_path}'")
    print(f" Best BEN-14K Checkpoint   : '{best_ben_path}'")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Standalone CFM Latent Bridge Dedicated Trainer")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v10/saber_unified_clean.pth", help="Path to pre-trained Master Encoder checkpoint")
    parser.add_argument("--data_dir", type=str, default="Datasets/benv1_14k", help="Path to BEN-14K dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training CFM bridge")
    parser.add_argument("--epochs", type=int, default=15, help="Number of CFM bridge training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for CFM bridge optimizer")
    parser.add_argument("--save_dir", type=str, default="checkpoints_v10", help="Directory to save trained bridge checkpoints")
    args = parser.parse_args()

    train_cfm_standalone(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()
