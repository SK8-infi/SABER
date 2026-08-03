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
    bridge_epochs_override: Optional[int] = None,
    mode: str = "all",
    synthetic_override: Optional[bool] = None
) -> None:
    print("="*80)
    print(" 🚀 UNIFIED SENSOR-AGNOSTIC SABER MASTER TRAINING ENGINE (SPEED OPTIMIZED)")
    print("="*80)

    config = load_config(config_path)
    if epochs_override is not None:
        config.train.epochs = epochs_override
    if synthetic_override is not None:
        config.dataset.use_synthetic = synthetic_override

    logger = setup_logger(name="saber_unified", log_dir=config.log_dir)
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    logger.info(f"Computation Device: {device} | Execution Mode: '{mode.upper()}' | CuDNN Benchmark: ACTIVE")

    # Load spatial data transforms (Full Multi-Crop: 2 Global 224x224 + 4 Local 96x96)
    train_transform = get_transforms(image_size=config.dataset.image_size, is_train=True, multi_crop=True)

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
    batch_size = batch_size_override or 48  # Fast Baseline Default: 48 (uses ~11.8 GB VRAM)

    ben14k_loader = DataLoader(
        ben14k_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    dsrsid_loader = DataLoader(
        dsrsid_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
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

    # Checkpoint output directories (Local & Google Drive 40 Epochs folder)
    local_40_dir = os.path.join(config.checkpoint_dir, "40epochs")
    os.makedirs(local_40_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    unified_ckpt_path = os.path.join(local_40_dir, "saber_unified.pth")
    latest_ckpt_path = os.path.join(local_40_dir, "latest.pth")
    unified_bridge_path = os.path.join(local_40_dir, "bridge_unified.pth")
    legacy_bridge_path = os.path.join(local_40_dir, "bridge_best.pth")

    # Google Drive Sync Directory Setup (Dynamic Folder based on checkpoint_dir)
    drive_data_dir = "/content/drive/MyDrive/SABER_Data"
    ckpt_dir_name = getattr(config, "checkpoint_dir", "checkpoints_sigreg").strip("/").replace("/", "_")
    if ckpt_dir_name == "checkpoints":
        drive_folder_name = "checkpoints_40epochs"
    else:
        drive_folder_name = ckpt_dir_name
    drive_ckpt_dir = os.path.join(drive_data_dir, drive_folder_name)
    is_drive_available = os.path.exists("/content/drive/MyDrive")
    if is_drive_available:
        os.makedirs(drive_ckpt_dir, exist_ok=True)

    mode_clean = mode.lower().strip()

    # -------------------------------------------------------------
    # PHASE 1: MASTER ENCODER & PROJECTION HEAD JOINT TRAINING
    # -------------------------------------------------------------
    if mode_clean in ["all", "encoder"]:
        # 4. Instantiate Pure Unsupervised Loss (Zero Classification Weight)
        loss_fn = SaberCombinedLoss(
            jaccard_weight=config.geometry.get("jaccard_weight", 2.0),
            ranking_weight=config.geometry.get("ranking_weight", 1.5),
            invariance_weight=config.loss.get("vicreg_invariance_weight", 15.0),
            variance_weight=config.loss.get("vicreg_variance_weight", 25.0),
            covariance_weight=config.loss.get("vicreg_covariance_weight", 2.0),
            sigreg_weight=config.geometry.get("sigreg_weight", 2.0),
            classification_weight=0.0,  # PURE UNSUPERVISED (NO LABELS)
            use_hyperbolic=getattr(model, "use_hyperbolic", False)
        ).to(device)

        # Optimizer (RiemannianAdam for geoopt Poincaré Ball or AdamW for Euclidean)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if getattr(model, "use_hyperbolic", False):
            try:
                import geoopt
                optimizer = geoopt.optim.RiemannianAdam(
                    trainable_params,
                    lr=config.train.get("learning_rate", 0.0003),
                    weight_decay=config.train.get("weight_decay", 0.01)
                )
                logger.info("Using geoopt.optim.RiemannianAdam optimizer for Hyperbolic Poincaré training.")
            except ImportError:
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=config.train.get("learning_rate", 0.0003),
                    weight_decay=config.train.get("weight_decay", 0.01)
                )
                logger.info("geoopt package not installed. Using native PyTorch AdamW for Hyperbolic training.")
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.train.get("learning_rate", 0.0003),
                weight_decay=config.train.get("weight_decay", 0.01)
            )

        use_amp = config.train.get("amp", True) and torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        epochs = epochs_override if epochs_override is not None else config.train.get("epochs", 5)
        grad_clip = config.train.get("grad_clip", 1.0)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # 🔍 Auto-Resume Detection for Phase 1 Encoder (Google Drive or Local)
        start_epoch = 1
        resume_candidates = [
            os.path.join(drive_ckpt_dir, "saber_unified.pth") if is_drive_available else None,
            os.path.join(drive_ckpt_dir, "latest.pth") if is_drive_available else None,
            unified_ckpt_path,
            latest_ckpt_path
        ]
        resume_path = None
        for candidate in resume_candidates:
            if candidate and os.path.exists(candidate):
                resume_path = candidate
                break

        if resume_path:
            try:
                logger.info(f"🔍 Found existing Master Checkpoint at '{resume_path}'. Auto-resuming Phase 1...")
                try:
                    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(resume_path, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                
                last_epoch = ckpt.get("epoch", 0)
                if last_epoch >= epochs:
                    logger.info(f"✅ Phase 1 Encoder training already completed ({last_epoch}/{epochs} epochs). Skipping to Phase 2!")
                    start_epoch = epochs + 1
                else:
                    start_epoch = last_epoch + 1
                    for _ in range(last_epoch):
                        scheduler.step()
                    logger.info(f"⏩ Auto-resuming Phase 1 Encoder training from Epoch {start_epoch}/{epochs}!")
            except Exception as e:
                logger.warning(f"Failed to auto-resume from '{resume_path}': {e}. Starting Phase 1 fresh.")

        logger.info("="*60)
        logger.info(f" PHASE 1: MASTER ENCODER JOINT TRAINING ({epochs} Epochs | SPEED OPTIMIZED)")
        logger.info("="*60)

        for epoch in range(start_epoch, epochs + 1):
            model.train()
            total_loss = 0.0
            sum_jacc, sum_rank, sum_inv, sum_var, sum_cov, sum_sigreg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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

                    if images_ben.shape[-1] != 224 or images_ben.shape[-2] != 224:
                        images_ben = F.interpolate(images_ben, size=(224, 224), mode="bilinear", align_corners=False)

                    x_s1 = images_ben[:, :2, :, :]
                    x_s2 = images_ben[:, 2:, :, :]

                    with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                        z1_ben, z2_ben, z1_pred_ben = model(x_s1, x_s2)[:3]
                        loss_dict = loss_fn(z1_ben, z2_ben, z1_pred_ben, targets=None)
                else:
                    try:
                        dsr_batch = next(dsr_iter)
                    except StopIteration:
                        dsr_iter = iter(dsrsid_loader)
                        dsr_batch = next(dsr_iter)

                    images_dsr = dsr_batch.get("image1", dsr_batch.get("image")).to(device, non_blocking=True)

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

                        loss_dict = loss_fn(z_pan, z_ms, z_pan_pred, targets=None)

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

                # Update EMA target model
                with torch.no_grad():
                    for p_online, p_target in zip(model.parameters(), ema_model.parameters()):
                        p_target.data.mul_(ema_decay).add_(p_online.data, alpha=1.0 - ema_decay)

                # Accumulate pure SSL loss sub-components
                v_loss = step_loss.item()
                v_jacc = loss_dict.get("jaccard_loss", torch.tensor(0.0)).item()
                v_rank = loss_dict.get("ranking_loss", torch.tensor(0.0)).item()
                v_inv = loss_dict.get("invariance_loss", torch.tensor(0.0)).item()
                v_var = loss_dict.get("variance_loss", torch.tensor(0.0)).item()
                v_cov = loss_dict.get("covariance_loss", torch.tensor(0.0)).item()
                v_sigreg = loss_dict.get("sigreg_loss", torch.tensor(0.0)).item()

                total_loss += v_loss
                sum_jacc += v_jacc
                sum_rank += v_rank
                sum_inv += v_inv
                sum_var += v_var
                sum_cov += v_cov
                sum_sigreg += v_sigreg

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{v_loss:.4f}",
                    "jacc": f"{v_jacc:.3f}",
                    "invar": f"{v_inv:.3f}",
                    "var": f"{v_var:.3f}",
                    "cov": f"{v_cov:.3f}",
                    "sigreg": f"{v_sigreg:.3f}",
                    "lr": f"{current_lr:.2e}"
                })

            scheduler.step()
            elapsed = time.time() - start_time
            avg_loss = total_loss / max_batches
            avg_jacc = sum_jacc / max_batches
            avg_rank = sum_rank / max_batches
            avg_inv = sum_inv / max_batches
            avg_var = sum_var / max_batches
            avg_cov = sum_cov / max_batches
            avg_sigreg = sum_sigreg / max_batches

            logger.info(
                f"Epoch [{epoch}/{epochs}] completed in {elapsed:.1f}s | "
                f"Loss: {avg_loss:.4f} | "
                f"Jacc: {avg_jacc:.4f} | "
                f"Rank: {avg_rank:.4f} | "
                f"Invar: {avg_inv:.4f} | "
                f"Var: {avg_var:.4f} | "
                f"Cov: {avg_cov:.4f} | "
                f"SigReg: {avg_sigreg:.4f}"
            )

            # Save Master Unified Checkpoint locally
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
            
            # Immediately sync checkpoint to Google Drive after each epoch
            if is_drive_available:
                try:
                    import shutil
                    shutil.copy2(unified_ckpt_path, os.path.join(drive_ckpt_dir, "saber_unified.pth"))
                    shutil.copy2(latest_ckpt_path, os.path.join(drive_ckpt_dir, "latest.pth"))
                    logger.info(f"💾 Synced Epoch [{epoch}/{epochs}] checkpoint to Google Drive: '{drive_ckpt_dir}'")
                except Exception as sync_e:
                    logger.warning(f"Google Drive sync warning: {sync_e}")

    # Clear VRAM cache between Phase 1 and Phase 2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING
    # -------------------------------------------------------------
    if mode_clean in ["all", "bridge"]:
        logger.info("="*60)
        logger.info(" PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING (SPEED OPTIMIZED)")
        logger.info("="*60)

        # If mode is bridge-only, ensure encoder weights are loaded
        if mode_clean == "bridge":
            encoder_path = resolve_existing_path(
                "",
                [
                    os.path.join(drive_ckpt_dir, "saber_unified.pth") if is_drive_available else "",
                    os.path.join(drive_ckpt_dir, "latest.pth") if is_drive_available else "",
                    unified_ckpt_path,
                    latest_ckpt_path
                ]
            )
            if encoder_path:
                logger.info(f"Loading master encoder weights from '{encoder_path}' for CFM bridge training...")
                try:
                    ckpt = torch.load(encoder_path, map_location=device, weights_only=False)
                except TypeError:
                    ckpt = torch.load(encoder_path, map_location=device)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
            else:
                logger.warning("⚠️ No pre-trained master encoder checkpoint found! CFM Bridge will train on initial backbone weights.")

        bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=4, dropout=0.1).to(device)
        bridge_opt = torch.optim.AdamW(bridge_net.parameters(), lr=0.0003, weight_decay=0.01)

        bridge_epochs = bridge_epochs_override if bridge_epochs_override is not None else 10
        model.eval()

        # 🔍 Auto-Resume Detection for Phase 2 CFM Bridge
        start_b_epoch = 1
        bridge_resume_candidates = [
            os.path.join(drive_ckpt_dir, "bridge_unified.pth") if is_drive_available else None,
            unified_bridge_path,
            legacy_bridge_path
        ]
        bridge_resume_path = None
        for b_cand in bridge_resume_candidates:
            if b_cand and os.path.exists(b_cand):
                bridge_resume_path = b_cand
                break

        if bridge_resume_path:
            try:
                logger.info(f"🔍 Found existing Bridge Checkpoint at '{bridge_resume_path}'. Auto-resuming Phase 2...")
                try:
                    b_ckpt = torch.load(bridge_resume_path, map_location=device, weights_only=False)
                except TypeError:
                    b_ckpt = torch.load(bridge_resume_path, map_location=device)
                if isinstance(b_ckpt, dict) and "bridge_state_dict" in b_ckpt:
                    bridge_net.load_state_dict(b_ckpt["bridge_state_dict"])
                    if "optimizer_state_dict" in b_ckpt:
                        bridge_opt.load_state_dict(b_ckpt["optimizer_state_dict"])
                    last_b_epoch = b_ckpt.get("epoch", 0)
                    if last_b_epoch >= bridge_epochs:
                        logger.info(f"✅ Phase 2 Bridge training already completed ({last_b_epoch}/{bridge_epochs} epochs).")
                        start_b_epoch = bridge_epochs + 1
                    else:
                        start_b_epoch = last_b_epoch + 1
                        logger.info(f"⏩ Auto-resuming Phase 2 Bridge training from Epoch {start_b_epoch}/{bridge_epochs}!")
                elif isinstance(b_ckpt, dict) and "state_dict" in b_ckpt:
                    bridge_net.load_state_dict(b_ckpt["state_dict"])
                elif isinstance(b_ckpt, dict):
                    bridge_net.load_state_dict(b_ckpt)
            except Exception as e:
                logger.warning(f"Failed to load bridge checkpoint from '{bridge_resume_path}': {e}")

        logger.info(f"Training Master CFM Bridge for {bridge_epochs} Epochs on Cross-Modal Pair Features...")

        for b_epoch in range(start_b_epoch, bridge_epochs + 1):
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

                # Flow Matching Interpolation
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
            logger.info(f"Phase 2 Bridge Epoch [{b_epoch}/{bridge_epochs}] | Flow Matching Loss: {avg_b_loss:.6f}")

            # Save local Bridge checkpoint
            b_payload = {
                "epoch": b_epoch,
                "bridge_state_dict": bridge_net.state_dict(),
                "optimizer_state_dict": bridge_opt.state_dict(),
                "loss": avg_b_loss
            }
            torch.save(b_payload, unified_bridge_path)
            torch.save(b_payload, legacy_bridge_path)

            # Immediately sync Bridge checkpoint to Google Drive after each epoch
            if is_drive_available:
                try:
                    import shutil
                    shutil.copy2(unified_bridge_path, os.path.join(drive_ckpt_dir, "bridge_unified.pth"))
                    shutil.copy2(legacy_bridge_path, os.path.join(drive_ckpt_dir, "bridge_best.pth"))
                    logger.info(f"💾 Synced Bridge Epoch [{b_epoch}/{bridge_epochs}] to Google Drive: '{drive_ckpt_dir}'")
                except Exception as sync_e:
                    logger.warning(f"Google Drive bridge sync warning: {sync_e}")

    # -------------------------------------------------------------
    # SAVE LIGHTWEIGHT CLEAN INFERENCE CHECKPOINT (STRIP OPTIMIZER & EMA BLOAT)
    # -------------------------------------------------------------
    clean_payload = {
        "model_state_dict": model.state_dict(),
        "config": config
    }
    clean_local_path = os.path.join(local_40_dir, "saber_unified_clean.pth")
    torch.save(clean_payload, clean_local_path)
    logger.info(f"🧹 Saved lightweight clean inference checkpoint to '{clean_local_path}'")

    if is_drive_available:
        try:
            import shutil
            shutil.copy2(clean_local_path, os.path.join(drive_ckpt_dir, "saber_unified_clean.pth"))
            logger.info(f"💾 Synced lightweight clean inference checkpoint (~340 MB) to Google Drive: '{drive_ckpt_dir}/saber_unified_clean.pth'")
        except Exception as sync_e:
            logger.warning(f"Google Drive clean checkpoint sync warning: {sync_e}")

    print("="*80)
    print(" 🎉 MASTER UNIFIED TRAINING COMPLETED SUCCESSFULLY (SPEED OPTIMIZED)!")
    print(f" Master Model Checkpoint (Full Resume): '{unified_ckpt_path}' (~1.05 GB)")
    print(f" Master Model Checkpoint (Clean Eval):  '{clean_local_path}' (~340 MB)")
    print(f" Master Bridge Checkpoint            : '{unified_bridge_path}' (~76 MB)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Train Unified Sensor-Agnostic SABER Engine & CFM Bridge (Speed Optimized)")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to BEN-14K dataset directory")
    parser.add_argument("--dsrsid_path", type=str, default=None, help="Path to DSRSID dataset mat file/dir")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training (default: 48 for ~11.8 GB VRAM utilization)")
    parser.add_argument("--epochs", type=int, default=None, help="Custom number of Phase 1 Encoder training epochs")
    parser.add_argument("--bridge_epochs", type=int, default=5, help="Custom number of Phase 2 CFM Bridge training epochs (default: 5)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "encoder", "bridge"], help="Training mode: 'all' (Encoder + Bridge), 'encoder' (Encoder only), 'bridge' (Bridge only)")
    parser.add_argument("--synthetic", type=str, default=None)
    args = parser.parse_args()

    synthetic_bool = (args.synthetic.lower() == "true") if args.synthetic is not None else None
    train_unified(
        config_path=args.config,
        data_dir_override=args.data_dir,
        dsrsid_path_override=args.dsrsid_path,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        bridge_epochs_override=args.bridge_epochs,
        mode=args.mode,
        synthetic_override=synthetic_bool
    )

if __name__ == "__main__":
    main()
