import sys
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.datasets.ben14k import BEN14KDataset
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge
from Saber.trainer.evaluator import Evaluator

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
    Original Pure Standalone Dedicated Training Engine for CFM Latent Bridge.
    
    1. Loads pre-trained Master Encoder (DOFA + LoRA + Projection Head).
    2. Freezes encoder completely.
    3. Trains CFM Bridge using pure Flow Matching MSE Loss:
       - Velocity Field Target: v_target = z2 - z1
       - Interpolated State: z_tau = (1 - tau) * z1 + tau * z2
       - Loss: MSELoss(v_pred, v_target)
    4. Evaluates Cross-Modal Retrieval (S1 Query -> S2 Gallery) after every epoch.
    5. Saves best bridge checkpoint based on Cross-Modal mAP@5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info(" 🚀 SABER ORIGINAL SIMPLE CFM LATENT BRIDGE TRAINING ENGINE")
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

    # 2. Instantiate Fresh Pure CFM Bridge
    bridge_net = CFMBridge(
        dim=768,
        hidden_dim=768,
        num_blocks=4,
        dropout=0.1
    ).to(device)

    # Bypass queries for clean direct concatenation
    bridge_net.is_queries_trained = False
    with torch.no_grad():
        if hasattr(bridge_net, "query_scale"):
            bridge_net.query_scale.zero_()

    optimizer = torch.optim.AdamW(bridge_net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Attach bridge to model for evaluator compatibility
    model.bridge.cfm_bridge = bridge_net
    model.bridge.ode_steps = 10

    # 3. Load Datasets
    logger.info(f"Loading BEN-14K dataset from '{data_dir}'...")
    train_dataset = BEN14KDataset(data_dir=data_dir, modality="both", split="train", is_train=True, use_synthetic=False)
    test_dataset = BEN14KDataset(data_dir=data_dir, modality="both", split="test", is_train=False, use_synthetic=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    evaluator = Evaluator(
        model=model,
        dataloader=test_loader,
        device=device,
        config=config
    )

    best_map5 = 0.0
    best_bridge_path = os.path.join(save_dir, "bridge_unified.pth")
    best_ben_path = os.path.join(save_dir, "bridge_best_ben14k.pth")

    logger.info("=" * 80)
    logger.info(" STARTING PURE CFM LATENT BRIDGE TRAINING & REAL-TIME EVALUATION")
    logger.info("=" * 80)

    for epoch in range(1, epochs + 1):
        bridge_net.train()
        total_loss = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"CFM Epoch [{epoch}/{epochs}]", leave=True, dynamic_ncols=True)
        for batch in pbar:
            images = batch.get("image1", batch.get("image")).to(device, non_blocking=True)
            if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

            x_s1 = images[:, :2, :, :]
            x_s2 = images[:, 2:, :, :]

            # Extract frozen 768-D embeddings directly from master model
            with torch.no_grad():
                z1_raw, z2_raw = model(x_s1, x_s2)[:2]
                z1 = F.normalize(z1_raw, p=2, dim=-1)
                z2 = F.normalize(z2_raw, p=2, dim=-1)

            optimizer.zero_grad()

            # Flow Matching Interpolation: z_tau = (1 - tau) * z1 + tau * z2
            B = z1.shape[0]
            tau = torch.rand(B, 1, device=device)
            z_tau = (1.0 - tau) * z1 + tau * z2
            v_target = z2 - z1

            # Predict velocity field
            v_pred, _ = bridge_net(z_tau, tau, z1)

            # Pure Flow Matching Velocity Field MSE Loss
            loss_bridge = F.mse_loss(v_pred, v_target)

            loss_bridge.backward()
            torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_bridge.item()

            pbar.set_postfix({
                "flow_mse": f"{loss_bridge.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()
        elapsed = time.time() - start_time
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches

        logger.info(
            f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | "
            f"Flow Matching MSE Loss: {avg_loss:.6f}"
        )

        # 4. Perform Real-Time Cross-Modal Evaluation (S1 -> S2)
        logger.info(f"📊 Evaluating Epoch [{epoch}/{epochs}] Cross-Modal Retrieval (S1 Query -> S2 Gallery)...")
        bridge_net.eval()
        eval_results = evaluator.evaluate()
        metrics = eval_results.get("metrics", {})
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
    print(" 🎉 PURE CFM LATENT BRIDGE TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best Cross-Modal mAP@5 : {best_map5:.4f}")
    print(f" Master Bridge Checkpoint : '{best_bridge_path}'")
    print(f" Best BEN-14K Checkpoint   : '{best_ben_path}'")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Original Pure Standalone CFM Latent Bridge Trainer")
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
