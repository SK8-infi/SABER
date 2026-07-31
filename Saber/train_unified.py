import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge, CFMBridgeWrapper
from Saber.losses.saber_loss import SaberCombinedLoss

def resolve_existing_path(path: str, candidate_paths: list) -> str:
    """Smart path resolver for Linux case-sensitive filesystems (Google Colab / Kaggle)."""
    if path and os.path.exists(path):
        return path
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return path or ""

def train_unified(
    config_path: str = "Saber/configs/config.yaml",
    data_dir_override: Optional[str] = None,
    dsrsid_path_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    synthetic_override: Optional[bool] = None,
    resume: bool = True
) -> None:
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

    # Resolve paths for Linux case sensitivity
    ben_raw_path = data_dir_override or config.dataset.data_dir
    ben_resolved_path = resolve_existing_path(
        ben_raw_path,
        [
            "Datasets/benv1_14k",
            "datasets/benv1_14k",
            "Datasets/ben14k",
            "datasets/ben14k",
            "/content/SABER/Datasets/benv1_14k"
        ]
    )

    dsr_raw_path = dsrsid_path_override or config.dataset.get("dsrsid_path", "datasets/DSRSID.mat")
    dsr_resolved_path = resolve_existing_path(
        dsr_raw_path,
        [
            "Datasets/DSRSID/DSRSID-001.mat",
            "Datasets/DSRSID/DSRSID.mat",
            "Datasets/DSRSID-001.mat",
            "Datasets/DSRSID.mat",
            "Datasets/DSRSID",
            "datasets/DSRSID.mat",
            "/content/SABER/Datasets/DSRSID/DSRSID-001.mat",
            "/content/SABER/Datasets/DSRSID/DSRSID.mat"
        ]
    )

    # 1. Dataset 1: BEN-14K (Sentinel-1 SAR 2ch + Sentinel-2 MS 12ch = 14ch)
    logger.info(f"Initializing BEN-14K Sentinel-1/2 dataset from '{ben_resolved_path}'...")
    ben14k_dataset = BEN14KDataset(
        data_dir=ben_resolved_path,
        use_synthetic=config.dataset.use_synthetic,
        size=config.dataset.get("size", 14832),
        image_size=config.dataset.image_size,
        transform=train_transform,
        modality="both",
        is_train=True,
        split="train"
    )

    # 2. Dataset 2: DSRSID (Gaofen PAN 1ch + Gaofen MS 4ch = 5ch, Stratified Balanced 14,000 samples)
    logger.info(f"Initializing DSRSID Gaofen PAN/MS dataset from '{dsr_resolved_path}'...")
    dsrsid_dataset = DSRSIDDataset(
        data_dir=dsr_resolved_path,
        use_synthetic=config.dataset.use_synthetic,
        size=14000,
        image_size=config.dataset.image_size,
        transform=train_transform,
        modality="both",
        is_train=True,
        split="train"
    )

    num_workers = config.dataset.get("num_workers", 2)
    batch_size = batch_size_override or config.dataset.get("batch_size", 64)

    ben14k_loader = DataLoader(
        ben14k_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )
    dsrsid_loader = DataLoader(
        dsrsid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=True
    )

    logger.info(f"BEN-14K Batches: {len(ben14k_loader)} | DSRSID Batches: {len(dsrsid_loader)} (Batch Size: {batch_size})")

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

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    unified_ckpt_path = os.path.join(config.checkpoint_dir, "saber_unified.pth")
    latest_ckpt_path = os.path.join(config.checkpoint_dir, "latest.pth")

    # -------------------------------------------------------------
    # SMART CHECKPOINT RESUME (Google Drive / Local)
    # -------------------------------------------------------------
    start_epoch = 1
    if resume:
        resume_candidates = [
            "/content/drive/MyDrive/SABER_Data/checkpoints/saber_unified.pth",
            "/content/drive/MyDrive/SABER_Data/checkpoints/latest.pth",
            unified_ckpt_path,
            latest_ckpt_path
        ]
        found_resume_path = resolve_existing_path("", resume_candidates)
        if found_resume_path:
            logger.info(f"🔄 Resuming Master Unified Training from existing checkpoint: '{found_resume_path}'...")
            try:
                ckpt = torch.load(found_resume_path, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                logger.info(f"✅ Successfully loaded state! Resuming from Epoch {start_epoch} -> {epochs}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load resume checkpoint ({e}). Starting fresh from Epoch 1.")

    # -------------------------------------------------------------
    # PHASE 1: MASTER ENCODER & PROJECTION HEAD JOINT TRAINING
    # -------------------------------------------------------------
    logger.info("="*60)
    logger.info(f" PHASE 1: MASTER ENCODER JOINT TRAINING ({start_epoch} -> {epochs} Epochs)")
    logger.info("="*60)

    if start_epoch <= epochs:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0
            sum_jacc, sum_rank, sum_inv, sum_var, sum_cov, sum_cls = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            start_time = time.time()

            ben_iter = iter(ben14k_loader)
            dsr_iter = iter(dsrsid_loader)
            max_batches = len(ben14k_loader) + len(dsrsid_loader)

            pbar = tqdm(range(max_batches), desc=f"Phase 1 Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True)

            for step in pbar:
                optimizer.zero_grad()
                is_ben_step = (step % 2 == 0)

                if is_ben_step:
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

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        z1_ben, z2_ben, z1_pred_ben, logits1_ben, logits2_ben = model(x_s1, x_s2)
                        loss_dict = loss_fn(z1_ben, z2_ben, z1_pred_ben, labels_ben, logits_s1=logits1_ben, logits_s2=logits2_ben)
                else:
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
                        feats_pan = model.backbone(x_pan, [0.675])
                        z_pan = model.projection_head(feats_pan)
                        z_pan_pred = model.predictor(z_pan)

                        feats_ms = model.backbone(x_ms, [0.485, 0.555, 0.660, 0.830])
                        z_ms = model.projection_head(feats_ms)

                        loss_dict = loss_fn(z_pan, z_ms, z_pan_pred, labels_dsr)

                step_loss = loss_dict.get("loss", loss_dict.get("total_loss"))

                if scaler is not None:
                    scaler.scale(step_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    optimizer.step()

                with torch.no_grad():
                    for p_online, p_target in zip(model.parameters(), ema_model.parameters()):
                        p_target.data.mul_(ema_decay).add_(p_online.data, alpha=1.0 - ema_decay)

                v_loss = step_loss.item()
                v_jacc = loss_dict.get("jaccard_loss", torch.tensor(0.0)).item()
                v_rank = loss_dict.get("ranking_loss", torch.tensor(0.0)).item()
                v_inv = loss_dict.get("invariance_loss", torch.tensor(0.0)).item()
                v_var = loss_dict.get("variance_loss", torch.tensor(0.0)).item()
                v_cov = loss_dict.get("covariance_loss", torch.tensor(0.0)).item()
                v_cls = loss_dict.get("classification_loss", torch.tensor(0.0)).item()

                total_loss += v_loss
                sum_jacc += v_jacc
                sum_rank += v_rank
                sum_inv += v_inv
                sum_var += v_var
                sum_cov += v_cov
                sum_cls += v_cls

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{v_loss:.4f}",
                    "jacc": f"{v_jacc:.3f}",
                    "invar": f"{v_inv:.3f}",
                    "var": f"{v_var:.3f}",
                    "cov": f"{v_cov:.3f}",
                    "bce": f"{v_cls:.3f}",
                    "lr": f"{current_lr:.2e}"
                })

            elapsed = time.time() - start_time
            avg_loss = total_loss / max_batches
            avg_jacc = sum_jacc / max_batches
            avg_rank = sum_rank / max_batches
            avg_inv = sum_inv / max_batches
            avg_var = sum_var / max_batches
            avg_cov = sum_cov / max_batches
            avg_cls = sum_cls / max_batches

            logger.info(
                f"Epoch [{epoch}/{epochs}] completed in {elapsed:.1f}s | "
                f"Loss: {avg_loss:.4f} | "
                f"Jacc: {avg_jacc:.4f} | "
                f"Rank: {avg_rank:.4f} | "
                f"Invar: {avg_inv:.4f} | "
                f"Var: {avg_var:.4f} | "
                f"Cov: {avg_cov:.4f} | "
                f"Class: {avg_cls:.4f}"
            )

            # Save Master Unified Checkpoints locally and copy to Google Drive if available
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": config
            }
            torch.save(checkpoint_payload, unified_ckpt_path)
            torch.save(checkpoint_payload, latest_ckpt_path)

            drive_ckpt_dir = "/content/drive/MyDrive/SABER_Data/checkpoints"
            if os.path.exists(drive_ckpt_dir):
                try:
                    torch.save(checkpoint_payload, os.path.join(drive_ckpt_dir, "saber_unified.pth"))
                    torch.save(checkpoint_payload, os.path.join(drive_ckpt_dir, "latest.pth"))
                    logger.info(f"✅ Auto-synced Epoch {epoch} checkpoint to Google Drive: '{drive_ckpt_dir}'")
                except Exception as e:
                    logger.warning(f"⚠️ Drive sync warning: {e}")

            logger.info(f"Saved Master Unified SABER checkpoint to '{unified_ckpt_path}' and '{latest_ckpt_path}'")
    else:
        logger.info(f"Phase 1 already complete ({start_epoch - 1} >= {epochs} epochs). Skipping to Phase 2.")

    # -------------------------------------------------------------
    # PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING
    # -------------------------------------------------------------
    logger.info("="*60)
    logger.info(" PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING")
    logger.info("="*60)

    bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=4, dropout=0.1).to(device)
    bridge_opt = torch.optim.AdamW(bridge_net.parameters(), lr=0.0003, weight_decay=0.01)

    unified_bridge_path = os.path.join(config.checkpoint_dir, "bridge_unified.pth")
    legacy_bridge_path = os.path.join(config.checkpoint_dir, "bridge_best.pth")

    # Auto-resume bridge if present
    drive_bridge_path = "/content/drive/MyDrive/SABER_Data/checkpoints/bridge_unified.pth"
    bridge_resume_path = resolve_existing_path("", [drive_bridge_path, unified_bridge_path, legacy_bridge_path])
    if resume and bridge_resume_path:
        try:
            logger.info(f"🔄 Resuming Master CFM Bridge from '{bridge_resume_path}'...")
            bridge_net.load_state_dict(torch.load(bridge_resume_path, map_location=device))
            logger.info("✅ Master CFM Bridge weights loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load bridge checkpoint ({e}). Training from scratch.")

    bridge_epochs = max(15, epochs * 3)
    model.eval()

    logger.info(f"Training Master CFM Bridge for {bridge_epochs} Epochs on Cross-Modal Pair Features...")

    for b_epoch in range(1, bridge_epochs + 1):
        bridge_net.train()
        total_bridge_loss = 0.0

        ben_iter = iter(ben14k_loader)
        dsr_iter = iter(dsrsid_loader)
        max_batches = len(ben14k_loader) + len(dsrsid_loader)

        pbar_b = tqdm(range(max_batches), desc=f"Phase 2 Bridge Epoch {b_epoch}/{bridge_epochs}", leave=True, dynamic_ncols=True)
        for step in pbar_b:
            bridge_opt.zero_grad()
            is_ben_step = (step % 2 == 0)

            if is_ben_step:
                try:
                    ben_batch = next(ben_iter)
                except StopIteration:
                    ben_iter = iter(ben14k_loader)
                    ben_batch = next(ben_iter)

                images_ben = ben_batch.get("image1", ben_batch.get("image")).to(device, non_blocking=True)
                if images_ben.shape[-1] != 224 or images_ben.shape[-2] != 224:
                    images_ben = F.interpolate(images_ben, size=(224, 224), mode="bilinear", align_corners=False)

                x1 = images_ben[:, :2, :, :]
                x2 = images_ben[:, 2:, :, :]
                wvs1, wvs2 = model.s1_wvs, model.s2_wvs
            else:
                try:
                    dsr_batch = next(dsr_iter)
                except StopIteration:
                    dsr_iter = iter(dsrsid_loader)
                    dsr_batch = next(dsr_iter)

                images_dsr = dsr_batch.get("image1", dsr_batch.get("image")).to(device, non_blocking=True)
                if images_dsr.shape[-1] != 224 or images_dsr.shape[-2] != 224:
                    images_dsr = F.interpolate(images_dsr, size=(224, 224), mode="bilinear", align_corners=False)

                x1 = images_dsr[:, :1, :, :]
                x2 = images_dsr[:, 1:, :, :]
                wvs1, wvs2 = [0.675], [0.485, 0.555, 0.660, 0.830]

            with torch.no_grad():
                feats1 = model.backbone(x1, wvs1)
                feats2 = model.backbone(x2, wvs2)
                z1 = model.projection_head(feats1)
                z2 = model.projection_head(feats2)

            tau = torch.rand(z1.shape[0], 1, device=device)
            z_tau = (1.0 - tau) * z1 + tau * z2
            target_velocity = z2 - z1

            v_pred, logvar = bridge_net(z_tau, tau, z1)
            b_loss = F.mse_loss(v_pred, target_velocity)

            b_loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
            bridge_opt.step()

            total_bridge_loss += b_loss.item()
            pbar_b.set_postfix({"flow_loss": f"{b_loss.item():.6f}"})

        avg_b_loss = total_bridge_loss / max_batches
        if b_epoch % 5 == 0 or b_epoch == bridge_epochs:
            logger.info(f"Phase 2 Bridge Epoch [{b_epoch}/{bridge_epochs}] | Flow Matching Loss: {avg_b_loss:.6f}")

    # Save Unified CFM Bridge Checkpoints locally & Drive
    torch.save(bridge_net.state_dict(), unified_bridge_path)
    torch.save(bridge_net.state_dict(), legacy_bridge_path)
    drive_ckpt_dir = "/content/drive/MyDrive/SABER_Data/checkpoints"
    if os.path.exists(drive_ckpt_dir):
        try:
            torch.save(bridge_net.state_dict(), os.path.join(drive_ckpt_dir, "bridge_unified.pth"))
            torch.save(bridge_net.state_dict(), os.path.join(drive_ckpt_dir, "bridge_best.pth"))
            logger.info(f"✅ Auto-synced CFM Bridge checkpoint to Google Drive!")
        except Exception as e:
            logger.warning(f"⚠️ Drive bridge sync warning: {e}")

    logger.info(f"Saved Master Unified CFM Bridge to '{unified_bridge_path}' and '{legacy_bridge_path}'")

    print("="*80)
    print(" 🎉 ALL MASTER UNIFIED TRAINING PHASES COMPLETE SUCCESSFULLY!")
    print(f" Master Model Checkpoint : '{unified_ckpt_path}'")
    print(f" Master Bridge Checkpoint: '{unified_bridge_path}'")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Train Unified Sensor-Agnostic SABER Engine & CFM Bridge")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to BEN-14K dataset directory")
    parser.add_argument("--dsrsid_path", type=str, default=None, help="Path to DSRSID dataset mat file/dir")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default="true", help="Resume from latest Google Drive checkpoint if available")
    parser.add_argument("--synthetic", type=str, default=None)
    args = parser.parse_args()

    synthetic_bool = (args.synthetic.lower() == "true") if args.synthetic is not None else None
    resume_bool = (args.resume.lower() == "true")
    train_unified(
        config_path=args.config,
        data_dir_override=args.data_dir,
        dsrsid_path_override=args.dsrsid_path,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        synthetic_override=synthetic_bool,
        resume=resume_bool
    )

if __name__ == "__main__":
    main()
