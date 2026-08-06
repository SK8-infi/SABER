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

import numpy as np
from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge, CFMBridgeWrapper
from Saber.losses.saber_loss import SaberCombinedLoss
from Saber.trainer.metrics import compute_retrieval_metrics

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
    synthetic_override: Optional[bool] = None,
    joint: bool = False,
    force_retrain_bridge: bool = False
) -> None:
    print("="*80)
    print(" 🚀 SABER MASTER TRAINING ENGINE (SPEED OPTIMIZED)")
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

    dataset_mode_str = "JOINT (BEN-14K + DSRSID)" if joint else "SINGLE DATASET (BEN-14K DEFAULT)"
    logger.info(f"Computation Device: {device} | Execution Mode: '{mode.upper()}' | Dataset Mode: '{dataset_mode_str}' | CuDNN Benchmark: ACTIVE")

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

    num_workers = config.dataset.get("num_workers", 2)
    batch_size = batch_size_override or 48  # Fast Baseline Default: 48 (uses ~11.8 GB VRAM)

    ben14k_loader = DataLoader(
        ben14k_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )

    dsrsid_loader = None
    if joint:
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
        dsrsid_loader = DataLoader(
            dsrsid_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        logger.info(f"Joint Mode ACTIVE: BEN-14K Batches: {len(ben14k_loader)} | DSRSID Batches: {len(dsrsid_loader)} (Batch Size: {batch_size})")
    else:
        logger.info(f"Single-Dataset Default Mode ACTIVE: BEN-14K Batches: {len(ben14k_loader)} (Batch Size: {batch_size}, DSRSID skipped)")

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

    # Google Drive Sync (Disabled - strictly operating 100% locally)
    is_drive_available = False
    drive_ckpt_dir = ""

    mode_clean = mode.lower().strip()

    # -------------------------------------------------------------
    # PHASE 1: MASTER ENCODER & PROJECTION HEAD JOINT TRAINING
    # -------------------------------------------------------------
    if mode_clean in ["all", "encoder"]:
        loss_fn = SaberCombinedLoss(
            jaccard_weight=config.geometry.get("jaccard_weight", 0.0),
            ranking_weight=config.geometry.get("ranking_weight", 0.0),
            invariance_weight=config.loss.get("vicreg_invariance_weight", 15.0),
            variance_weight=config.loss.get("vicreg_variance_weight", 25.0),
            covariance_weight=config.loss.get("vicreg_covariance_weight", 2.0),
            sigreg_weight=config.geometry.get("sigreg_weight", 2.0),
            classification_weight=0.0,  # PURE UNSUPERVISED (NO LABELS)
            infonce_weight=config.geometry.get("infonce_weight", 0.0),
            infonce_temperature=config.geometry.get("infonce_temperature", 0.07)
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

        epochs = epochs_override if epochs_override is not None else config.train.get("epochs", 5)
        grad_clip = config.train.get("grad_clip", 1.0)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        # 🔍 Auto-Resume Detection for Phase 1 Encoder (Strictly Local)
        start_epoch = 1
        resume_candidates = [
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
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
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
            sum_jacc, sum_rank, sum_inv, sum_var, sum_cov, sum_sigreg, sum_infonce = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            start_time = time.time()

            ben_iter = iter(ben14k_loader)
            dsr_iter = iter(dsrsid_loader) if (joint and dsrsid_loader is not None) else None
            max_batches = len(ben14k_loader) + (len(dsrsid_loader) if (joint and dsrsid_loader is not None) else 0)

            pbar = tqdm(range(max_batches), desc=f"Phase 1 Epoch {epoch}/{epochs}", leave=True, dynamic_ncols=True)

            for step in pbar:
                optimizer.zero_grad()
                is_ben_step = not joint or (step % 2 == 0) or (dsr_iter is None)

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
                accum_steps = config.train.get("grad_accumulation_steps", 1)
                loss_scaled = step_loss / accum_steps

                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                    if (step + 1) % accum_steps == 0 or (step + 1) == max_batches:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss_scaled.backward()
                    if (step + 1) % accum_steps == 0 or (step + 1) == max_batches:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                        optimizer.step()
                        optimizer.zero_grad()

                # Update EMA target model
                if (step + 1) % accum_steps == 0 or (step + 1) == max_batches:
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
                v_infonce = loss_dict.get("infonce_loss", torch.tensor(0.0)).item()

                total_loss += v_loss
                sum_jacc += v_jacc
                sum_rank += v_rank
                sum_inv += v_inv
                sum_var += v_var
                sum_cov += v_cov
                sum_sigreg += v_sigreg
                sum_infonce += v_infonce

                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{v_loss:.4f}",
                    "infonce": f"{v_infonce:.3f}",
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
            avg_infonce = sum_infonce / max_batches

            logger.info(
                f"Epoch [{epoch}/{epochs}] completed in {elapsed:.1f}s | "
                f"Loss: {avg_loss:.4f} | "
                f"InfoNCE: {avg_infonce:.4f} | "
                f"Jacc: {avg_jacc:.4f} | "
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
        torch.cuda.empty_cache()    # -------------------------------------------------------------
    # PHASE 2: MASTER PATCH-PROJECTED CFM BRIDGE TRAINING
    # -------------------------------------------------------------
    if mode_clean in ["all", "bridge"]:
        logger.info("="*60)
        logger.info(" PHASE 2: MASTER LATENT CFM BRIDGE TRAINING & INSTANT GPU EVALUATION")
        logger.info("="*60)

        # If mode is bridge-only, ensure encoder weights are loaded
        if mode_clean == "bridge":
            encoder_path = resolve_existing_path(
                "",
                [
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
                    model.load_state_dict(ckpt["model_state_dict"], strict=False)
            else:
                logger.warning("⚠️ No pre-trained master encoder checkpoint found! CFM Bridge will train on initial backbone weights.")

        # Freeze encoder during bridge training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=4, dropout=0.1).to(device)
        
        # Bypass queries for clean direct concatenation
        bridge_net.is_queries_trained = False
        with torch.no_grad():
            if hasattr(bridge_net, "query_scale"):
                bridge_net.query_scale.zero_()

        bridge_opt = torch.optim.AdamW(bridge_net.parameters(), lr=0.0005, weight_decay=0.01)

        bridge_epochs = bridge_epochs_override if bridge_epochs_override is not None else 15
        bridge_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(bridge_opt, T_max=bridge_epochs, eta_min=1e-6)

        # Attach bridge to model for ODE evaluation
        model.bridge.cfm_bridge = bridge_net
        model.bridge.ode_steps = 10

        # ⚡ PRE-EXTRACT TRAIN & TEST LATENTS ONCE INTO GPU MEMORY
        logger.info("⚡ Pre-extracting 768-D S1 (z1) & S2 (z2) latents for TRAIN set into GPU memory...")
        train_z1_list, train_z2_list = [], []
        with torch.no_grad():
            for batch in tqdm(ben14k_loader, desc="Caching Train Latents", dynamic_ncols=True):
                images = batch.get("image1", batch.get("image")).to(device, non_blocking=True)
                if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

                x_s1 = images[:, :2, :, :]
                x_s2 = images[:, 2:, :, :]

                z1_raw, z2_raw = model(x_s1, x_s2)[:2]
                z1_norm = F.normalize(z1_raw, p=2, dim=-1)
                z2_norm = F.normalize(z2_raw, p=2, dim=-1)

                train_z1_list.append(z1_norm.cpu())
                train_z2_list.append(z2_norm.cpu())

        cached_train_z1 = torch.cat(train_z1_list, dim=0).to(device)
        cached_train_z2 = torch.cat(train_z2_list, dim=0).to(device)
        N_train = cached_train_z1.shape[0]

        logger.info("⚡ Pre-extracting 768-D S1 (z1) & S2 (z2) latents for TEST evaluation set into GPU memory...")
        test_dataset = BEN14KDataset(data_dir=ben_resolved_path, modality="both", split="test", is_train=False, use_synthetic=False)
        test_loader = DataLoader(test_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        test_s1_list, test_s2_list, test_labels_list, test_names_list = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Caching Test Latents", dynamic_ncols=True):
                images = batch.get("image", batch.get("image1")).to(device, non_blocking=True)
                if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

                x_s1 = images[:, :2, :, :]
                x_s2 = images[:, 2:, :, :]

                z1_raw, z2_raw = model(x_s1, x_s2)[:2]
                z1_norm = F.normalize(z1_raw, p=2, dim=-1)
                z2_norm = F.normalize(z2_raw, p=2, dim=-1)

                test_s1_list.append(z1_norm.cpu())
                test_s2_list.append(z2_norm.cpu())
                test_labels_list.append(batch["label"].cpu())
                test_names_list.extend(batch["name"])

        cached_test_s1 = torch.cat(test_s1_list, dim=0).to(device)
        cached_test_s2 = torch.cat(test_s2_list, dim=0).to(device)
        cached_test_labels = torch.cat(test_labels_list, dim=0).to(device)
        cached_test_names = np.array(test_names_list)
        N_test = cached_test_s1.shape[0]

        # Query (20%) and Gallery (80%) split
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(N_test)
        q_size = max(1, N_test // 5)
        query_indices = np.sort(shuffled[:q_size])
        gallery_indices = np.sort(shuffled[q_size:])

        test_q_s1 = cached_test_s1[query_indices]
        test_q_labels = cached_test_labels[query_indices].cpu().numpy()
        test_q_names = cached_test_names[query_indices]

        test_g_s2 = cached_test_s2[gallery_indices]
        test_g_labels = cached_test_labels[gallery_indices].cpu().numpy()
        test_g_names = cached_test_names[gallery_indices]

        logger.info(f"✅ Pre-extraction Complete! Cached {N_train} train pairs & {N_test} test pairs in GPU memory.")
        logger.info(f"📊 Test Evaluation Setup: {len(query_indices)} S1 Queries -> {len(gallery_indices)} S2 Gallery Items.")

        best_f1_5 = 0.0
        best_map5 = 0.0
        best_ben_path = os.path.join(config.checkpoint_dir, "bridge_best_ben14k.pth")

        logger.info(f"Training Master CFM Bridge for {bridge_epochs} Epochs on Cached GPU Memory Tensors...")

        batch_size_b = config.dataset.batch_size
        num_batches_b = (N_train + batch_size_b - 1) // batch_size_b

        for b_epoch in range(1, bridge_epochs + 1):
            bridge_net.train()
            total_bridge_loss = 0.0
            start_b_time = time.time()

            perm = torch.randperm(N_train, device=device)

            for b_idx in range(num_batches_b):
                idx_b = perm[b_idx * batch_size_b : (b_idx + 1) * batch_size_b]
                z1_b = cached_train_z1[idx_b]
                z2_b = cached_train_z2[idx_b]
                B_curr = z1_b.shape[0]

                bridge_opt.zero_grad()

                tau = torch.rand(B_curr, 1, device=device)
                z_tau = (1.0 - tau) * z1_b + tau * z2_b
                v_target = z2_b - z1_b

                v_pred, _ = bridge_net(z_tau, tau, z1_b)
                loss_b = F.mse_loss(v_pred, v_target)

                loss_b.backward()
                torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
                bridge_opt.step()

                total_bridge_loss += loss_b.item()

            bridge_scheduler.step()
            train_b_elapsed = time.time() - start_b_time
            avg_b_loss = total_bridge_loss / num_batches_b

            # 📊 INSTANT GPU CROSS-MODAL EVALUATION (S1 -> S2)
            eval_b_start = time.time()
            bridge_net.eval()
            with torch.no_grad():
                translated_q_s1 = model.bridge(test_q_s1)
                translated_q_s1 = F.normalize(translated_q_s1, p=2, dim=-1)

                metrics5 = compute_retrieval_metrics(
                    query_embeds=translated_q_s1.cpu().numpy(),
                    gallery_embeds=test_g_s2.cpu().numpy(),
                    query_labels=test_q_labels,
                    gallery_labels=test_g_labels,
                    top_k=5,
                    is_multilabel=True,
                    query_names=test_q_names,
                    gallery_names=test_g_names,
                    exclude_self_matches=False
                )
                map5 = metrics5.get("map@5", metrics5.get("MAP@5", 0.0))
                prec5 = metrics5.get("precision@5", metrics5.get("PRECISION@5", 0.0))
                rec5 = metrics5.get("recall@5", metrics5.get("RECALL@5", 0.0))
                f1_5 = metrics5.get("f1@5", metrics5.get("F1@5", (2.0 * prec5 * rec5) / (prec5 + rec5 + 1e-8)))

            eval_b_elapsed = time.time() - eval_b_start

            logger.info(
                f"Phase 2 Bridge Epoch [{b_epoch}/{bridge_epochs}] ({train_b_elapsed:.2f}s train, {eval_b_elapsed:.2f}s eval) | "
                f"Flow MSE: {avg_b_loss:.6f} | F1@5: {f1_5:.4f} | mAP@5: {map5:.4f} | Prec@5: {prec5:.4f} | Rec@5: {rec5:.4f}"
            )

            # Save Bridge Checkpoints
            b_ckpt_data = {
                "epoch": b_epoch,
                "bridge_state_dict": bridge_net.state_dict(),
                "optimizer_state_dict": bridge_opt.state_dict(),
                "loss": avg_b_loss,
                "f1_5": f1_5,
                "map5": map5,
                "precision5": prec5,
                "recall5": rec5
            }
            torch.save(b_ckpt_data, unified_bridge_path)

            if f1_5 > best_f1_5:
                best_f1_5 = f1_5
                best_map5 = map5
                torch.save(b_ckpt_data, best_ben_path)
                logger.info(f"🏆 NEW BEST Cross-Modal F1@5: {best_f1_5:.4f} (mAP@5: {map5:.4f})! Saved to '{best_ben_path}'")

            if is_drive_available:
                try:
                    drive_b_path = os.path.join(drive_ckpt_dir, "bridge_unified.pth")
                    torch.save(b_ckpt_data, drive_b_path)
                    logger.info(f"💾 Synced Bridge checkpoint to Google Drive: '{drive_b_path}'")
                except Exception as sync_e:
                    logger.warning(f"Google Drive Bridge checkpoint sync warning: {sync_e}")

    # Save final clean inference checkpoints
    clean_local_path = os.path.join(config.checkpoint_dir, "saber_unified_clean.pth")
    clean_state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    clean_ckpt = {
        "architecture": getattr(config.model, "architecture", "saber"),
        "model_state_dict": clean_state_dict,
        "use_hyperbolic": getattr(config.model, "use_hyperbolic", False),
        "in_channels": 14,
        "epoch": epochs if mode_clean in ["all", "encoder"] else 0
    }
    torch.save(clean_ckpt, clean_local_path)
    logger.info(f"Saved lightweight clean inference model checkpoint (~340 MB) to '{clean_local_path}'")

    if is_drive_available:
        try:
            drive_clean_path = os.path.join(drive_ckpt_dir, "saber_unified_clean.pth")
            torch.save(clean_ckpt, drive_clean_path)
            logger.info(f"💾 Synced lightweight clean inference checkpoint (~340 MB) to Google Drive: '{drive_clean_path}'")
        except Exception as sync_e:
            logger.warning(f"Google Drive clean checkpoint sync warning: {sync_e}")

    print("="*80)
    print(" 🎉 MASTER TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Master Model Checkpoint (Full Resume): '{unified_ckpt_path}' (~1.05 GB)")
    print(f" Master Model Checkpoint (Clean Eval):  '{clean_local_path}' (~340 MB)")
    print(f" Master Bridge Checkpoint            : '{unified_bridge_path}' (~76 MB)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Train SABER Engine & CFM Bridge (Speed Optimized)")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to BEN-14K dataset directory")
    parser.add_argument("--dsrsid_path", type=str, default=None, help="Path to DSRSID dataset mat file/dir")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training (default: 48 for ~11.8 GB VRAM utilization)")
    parser.add_argument("--epochs", type=int, default=None, help="Custom number of Phase 1 Encoder training epochs")
    parser.add_argument("--bridge_epochs", type=int, default=5, help="Custom number of Phase 2 CFM Bridge training epochs (default: 5)")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "encoder", "bridge"], help="Training mode: 'all' (Encoder + Bridge), 'encoder' (Encoder only), 'bridge' (Bridge only)")
    parser.add_argument("--synthetic", type=str, default=None)
    parser.add_argument("--joint", action="store_true", help="Enable joint multi-dataset training (BEN-14K + DSRSID). Default: False (BEN-14K only)")
    parser.add_argument("--force_retrain_bridge", action="store_true", help="Force retraining Phase 2 CFM Bridge from scratch even if checkpoint exists")
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
        synthetic_override=synthetic_bool,
        joint=args.joint,
        force_retrain_bridge=args.force_retrain_bridge
    )

if __name__ == "__main__":
    main()
