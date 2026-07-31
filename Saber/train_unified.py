import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.utils.checkpoint import save_checkpoint
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge, CFMBridgeWrapper
from Saber.losses.saber_loss import SaberCombinedLoss

def train_unified(config_path: str = "Saber/configs/config.yaml", epochs_override: int = None, synthetic_override: bool = None) -> None:
    print("="*80)
    print(" 🚀 UNIFIED SENSOR-AGNOSTIC SABER MASTER TRAINING ENGINE")
    print("="*80)

    config = load_config(config_path)
    if epochs_override is not None:
        config.train.epochs = epochs_override
    if synthetic_override is not None:
        config.dataset.use_synthetic = synthetic_override

    logger = setup_logger(name="saber_unified", log_dir=config.log_dir)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load spatial data transforms
    train_transform = get_transforms(image_size=config.dataset.image_size, is_train=True)

    # 1. Dataset 1: BEN-14K (Sentinel-1 SAR 2ch + Sentinel-2 MS 12ch = 14ch)
    logger.info("Initializing BEN-14K Sentinel-1/2 dataset (14,832 samples)...")
    ben14k_dataset = BEN14KDataset(
        data_dir=config.dataset.data_dir,
        use_synthetic=config.dataset.use_synthetic,
        size=config.dataset.get("size", 14832),
        image_size=config.dataset.image_size,
        transform=train_transform,
        modality="both",
        is_train=True,
        split="train"
    )

    # 2. Dataset 2: DSRSID (Gaofen PAN 1ch + Gaofen MS 4ch = 5ch, Stratified Balanced 14,000 samples)
    logger.info("Initializing DSRSID Gaofen PAN/MS dataset (Stratified 14,000 samples)...")
    dsrsid_dataset = DSRSIDDataset(
        data_dir=config.dataset.get("dsrsid_path", "datasets/DSRSID.mat"),
        use_synthetic=config.dataset.use_synthetic,
        size=14000,
        image_size=config.dataset.image_size,
        transform=train_transform,
        modality="both",
        is_train=True,
        split="train"
    )

    num_workers = config.dataset.get("num_workers", 2)
    batch_size = config.dataset.get("batch_size", 32)

    ben14k_loader = DataLoader(
        ben14k_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    dsrsid_loader = DataLoader(
        dsrsid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )

    logger.info(f"BEN-14K Batches: {len(ben14k_loader)} | DSRSID Batches: {len(dsrsid_loader)}")

    # 3. Instantiate SINGLE UNIFIED SABER MODEL (DOFA ViT + LoRA + Shared Projection)
    model = SABER(config=config, in_channels=14).to(device)

    # Instantiate EMA target model
    ema_model = SABER(config=config, in_channels=14).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_decay = config.train.get("ema_decay", 0.996)

    # 4. Instantiate Unified Jaccard + VICReg Loss
    loss_fn = SaberCombinedLoss(
        jaccard_weight=config.geometry.get("jaccard_weight", 2.0),
        ranking_weight=config.geometry.get("ranking_weight", 1.5),
        invariance_weight=config.loss.get("vicreg_invariance_weight", 15.0),
        variance_weight=config.loss.get("vicreg_variance_weight", 25.0),
        covariance_weight=config.loss.get("vicreg_covariance_weight", 2.0),
        classification_weight=config.geometry.get("classification_weight", 1.0)
    ).to(device)

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.get("learning_rate", 0.0005),
        weight_decay=config.train.get("weight_decay", 0.01)
    )

    use_amp = config.train.get("amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    epochs = config.train.get("epochs", 5)
    grad_clip = config.train.get("grad_clip", 1.0)

    # -------------------------------------------------------------
    # PHASE 1: MASTER ENCODER & PROJECTION HEAD JOINT TRAINING
    # -------------------------------------------------------------
    logger.info("="*60)
    logger.info(f" PHASE 1: MASTER ENCODER JOINT TRAINING ({epochs} Epochs)")
    logger.info("="*60)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    unified_ckpt_path = os.path.join(config.checkpoint_dir, "saber_unified.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        ben_iter = iter(ben14k_loader)
        dsr_iter = iter(dsrsid_loader)
        max_batches = max(len(ben14k_loader), len(dsrsid_loader))

        for batch_idx in range(max_batches):
            optimizer.zero_grad()

            # Process BEN-14K Batch (S1 SAR 2ch <-> S2 MS 12ch)
            try:
                ben_batch = next(ben_iter)
            except StopIteration:
                ben_iter = iter(ben14k_loader)
                ben_batch = next(ben_iter)

            images_ben = ben_batch.get("image1", ben_batch.get("image")).to(device, non_blocking=True)
            labels_ben = ben_batch["label"].to(device, non_blocking=True)

            if images_ben.shape[-1] != 224 or images_ben.shape[-2] != 224:
                images_ben = F.interpolate(images_ben, size=(224, 224), mode="bilinear", align_corners=False)

            x_s1 = images_ben[:, :2, :, :]
            x_s2 = images_ben[:, 2:, :, :]

            # Process DSRSID Batch (Gaofen PAN 1ch <-> Gaofen MS 4ch)
            try:
                dsr_batch = next(dsr_iter)
            except StopIteration:
                dsr_iter = iter(dsrsid_loader)
                dsr_batch = next(dsr_iter)

            images_dsr = dsr_batch.get("image1", dsr_batch.get("image")).to(device, non_blocking=True)
            labels_dsr = dsr_batch["label"].to(device, non_blocking=True)

            if images_dsr.shape[-1] != 224 or images_dsr.shape[-2] != 224:
                images_dsr = F.interpolate(images_dsr, size=(224, 224), mode="bilinear", align_corners=False)

            x_pan = images_dsr[:, :1, :, :]
            x_ms = images_dsr[:, 1:, :, :]

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                # 1. Forward BEN-14K (S1/S2)
                z1_ben, z2_ben, z1_pred_ben, logits1_ben, logits2_ben = model(x_s1, x_s2)
                loss_dict_ben = loss_fn(z1_ben, z2_ben, z1_pred_ben, labels_ben, logits_s1=logits1_ben, logits_s2=logits2_ben)

                # 2. Forward DSRSID (PAN/MS)
                feats_pan = model.backbone(x_pan, [0.675])
                z_pan = model.projection_head(feats_pan)
                z_pan_pred = model.predictor(z_pan)

                feats_ms = model.backbone(x_ms, [0.485, 0.555, 0.660, 0.830])
                z_ms = model.projection_head(feats_ms)

                loss_dict_dsr = loss_fn(z_pan, z_ms, z_pan_pred, labels_dsr)

                # Aggregated Joint Loss
                batch_loss = loss_dict_ben["total_loss"] + loss_dict_dsr["total_loss"]

            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()

            # Update EMA target model
            with torch.no_grad():
                for p_online, p_target in zip(model.parameters(), ema_model.parameters()):
                    p_target.data.mul_(ema_decay).add_(p_online.data, alpha=1.0 - ema_decay)

            total_loss += batch_loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == max_batches:
                logger.info(f"Phase 1 Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{max_batches}] | Joint Loss: {batch_loss.item():.4f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max_batches
        logger.info(f"=== Phase 1 Epoch {epoch}/{epochs} Complete | Avg Joint Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ===")

        # Save Master Unified Checkpoint
        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config
            },
            filename=unified_ckpt_path
        )
        logger.info(f"Saved Master Unified SABER checkpoint to '{unified_ckpt_path}'")

    # -------------------------------------------------------------
    # PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING
    # -------------------------------------------------------------
    logger.info("="*60)
    logger.info(" PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING")
    logger.info("="*60)

    bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=4, dropout=0.1).to(device)
    bridge_opt = torch.optim.AdamW(bridge_net.parameters(), lr=0.0003, weight_decay=0.01)

    bridge_epochs = max(15, epochs * 3)
    model.eval()

    unified_bridge_path = os.path.join(config.checkpoint_dir, "bridge_unified.pth")
    legacy_bridge_path = os.path.join(config.checkpoint_dir, "bridge_best.pth")

    logger.info(f"Training Master CFM Bridge for {bridge_epochs} Epochs on Cross-Modal Pair Features...")

    for b_epoch in range(1, bridge_epochs + 1):
        bridge_net.train()
        total_bridge_loss = 0.0

        ben_iter = iter(ben14k_loader)
        dsr_iter = iter(dsrsid_loader)
        max_batches = max(len(ben14k_loader), len(dsrsid_loader))

        for batch_idx in range(max_batches):
            bridge_opt.zero_grad()

            try:
                ben_batch = next(ben_iter)
            except StopIteration:
                ben_iter = iter(ben14k_loader)
                ben_batch = next(ben_iter)

            images_ben = ben_batch.get("image1", ben_batch.get("image")).to(device, non_blocking=True)
            if images_ben.shape[-1] != 224 or images_ben.shape[-2] != 224:
                images_ben = F.interpolate(images_ben, size=(224, 224), mode="bilinear", align_corners=False)

            x_s1 = images_ben[:, :2, :, :]
            x_s2 = images_ben[:, 2:, :, :]

            with torch.no_grad():
                feats_s1 = model.backbone(x_s1, model.s1_wvs)
                feats_s2 = model.backbone(x_s2, model.s2_wvs)
                z_s1 = model.s1_projection(feats_s1)
                z_s2 = model.s2_projection(feats_s2)

            # Flow Matching Interpolation
            tau = torch.rand(z_s1.shape[0], 1, device=device)
            z_tau = (1.0 - tau) * z_s1 + tau * z_s2
            target_velocity = z_s2 - z_s1

            v_pred, logvar = bridge_net(z_tau, tau, z_s1)
            b_loss = F.mse_loss(v_pred, target_velocity)

            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
            bridge_opt.step()

            total_bridge_loss += b_loss.item()

        avg_b_loss = total_bridge_loss / max_batches
        if b_epoch % 5 == 0 or b_epoch == bridge_epochs:
            logger.info(f"Phase 2 Bridge Epoch [{b_epoch}/{bridge_epochs}] | Flow Matching Loss: {avg_b_loss:.6f}")

    # Save Unified CFM Bridge Checkpoints
    torch.save(bridge_net.state_dict(), unified_bridge_path)
    torch.save(bridge_net.state_dict(), legacy_bridge_path)
    logger.info(f"Saved Master Unified CFM Bridge to '{unified_bridge_path}' and '{legacy_bridge_path}'")

    print("="*80)
    print(" 🎉 ALL MASTER UNIFIED TRAINING PHASES COMPLETE SUCCESSFULLY!")
    print(f" Master Model Checkpoint : '{unified_ckpt_path}'")
    print(f" Master Bridge Checkpoint: '{unified_bridge_path}'")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Train Unified Sensor-Agnostic SABER Engine & CFM Bridge")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthetic", type=str, default=None)
    args = parser.parse_args()

    synthetic_bool = (args.synthetic.lower() == "true") if args.synthetic is not None else None
    train_unified(config_path=args.config, epochs_override=args.epochs, synthetic_override=synthetic_bool)

if __name__ == "__main__":
    main()
