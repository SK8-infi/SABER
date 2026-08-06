import sys
import os
import time
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.datasets.ben14k import BEN14KDataset
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge
from Saber.trainer.metrics import compute_retrieval_metrics

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
    Ultra-Fast Standalone CFM Latent Bridge Training & Instant GPU Evaluation Engine.
    
    1. Loads pre-trained Master Encoder (DOFA + LoRA + Projection Head).
    2. Pre-extracts 768-D S1 (z1) & S2 (z2) latents for TRAIN (10,382) & TEST (2,967) sets ONCE.
    3. Trains CFM Bridge directly on GPU memory tensors (~0.2s / epoch).
    4. Evaluates Cross-Modal Retrieval (S1 Query -> S2 Gallery) directly on GPU (~0.01s / evaluation).
    5. Saves best bridge checkpoint based on Cross-Modal mAP@5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info(" 🚀 SABER ULTRA-FAST CFM BRIDGE TRAINER & INSTANT GPU EVALUATOR")
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

    # Attach bridge to model for evaluation ODE integration
    model.bridge.cfm_bridge = bridge_net
    model.bridge.ode_steps = 10

    # 3. Load Datasets
    logger.info(f"Loading BEN-14K dataset from '{data_dir}'...")
    train_dataset = BEN14KDataset(data_dir=data_dir, modality="both", split="train", is_train=True, use_synthetic=False)
    test_dataset = BEN14KDataset(data_dir=data_dir, modality="both", split="test", is_train=False, use_synthetic=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 4. PRE-EXTRACT TRAIN & TEST LATENTS ONCE INTO GPU MEMORY
    logger.info("⚡ Pre-extracting 768-D S1 (z1) & S2 (z2) latents for TRAIN set into GPU memory...")
    train_z1_list, train_z2_list = [], []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Caching Train Latents", dynamic_ncols=True):
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

    # Pre-select evaluation query (20%) and gallery (80%) indices with fixed seed 42
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

    best_map5 = 0.0
    best_bridge_path = os.path.join(save_dir, "bridge_unified.pth")
    best_ben_path = os.path.join(save_dir, "bridge_best_ben14k.pth")

    logger.info("=" * 80)
    logger.info(" STARTING LIGHTNING-FAST CFM LATENT BRIDGE OPTIMIZATION")
    logger.info("=" * 80)

    for epoch in range(1, epochs + 1):
        bridge_net.train()
        total_loss = 0.0
        start_time = time.time()

        # Randomize batch order every epoch
        perm = torch.randperm(N_train, device=device)
        num_batches = (N_train + batch_size - 1) // batch_size

        for b_idx in range(num_batches):
            idx_batch = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
            z1_b = cached_train_z1[idx_batch]
            z2_b = cached_train_z2[idx_batch]
            B_curr = z1_b.shape[0]

            optimizer.zero_grad()

            # Flow Matching Interpolation: z_tau = (1 - tau) * z1 + tau * z2
            tau = torch.rand(B_curr, 1, device=device)
            z_tau = (1.0 - tau) * z1_b + tau * z2_b
            v_target = z2_b - z1_b

            # Predict velocity field
            v_pred, _ = bridge_net(z_tau, tau, z1_b)

            # Pure Flow Matching Velocity Field MSE Loss
            loss_bridge = F.mse_loss(v_pred, v_target)

            loss_bridge.backward()
            torch.nn.utils.clip_grad_norm_(bridge_net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss_bridge.item()

        scheduler.step()
        train_elapsed = time.time() - start_time
        avg_loss = total_loss / num_batches

        # 5. INSTANT GPU CROSS-MODAL EVALUATION (S1 -> S2)
        eval_start = time.time()
        bridge_net.eval()
        with torch.no_grad():
            # Translate S1 query vectors via CFM Bridge ODE integration
            translated_q_s1 = model.bridge(test_q_s1)
            translated_q_s1 = F.normalize(translated_q_s1, p=2, dim=-1)

            # Direct GPU metrics computation
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

        eval_elapsed = time.time() - eval_start

        logger.info(
            f"Epoch [{epoch}/{epochs}] ({train_elapsed:.2f}s train, {eval_elapsed:.2f}s eval) | "
            f"Flow MSE: {avg_loss:.6f} | mAP@5: {map5:.4f} | Prec@5: {prec5:.4f} | Rec@5: {rec5:.4f}"
        )

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
    print(" 🎉 INSTANT CFM LATENT BRIDGE TRAINING COMPLETED SUCCESSFULLY!")
    print(f" Best Cross-Modal mAP@5 : {best_map5:.4f}")
    print(f" Master Bridge Checkpoint : '{best_bridge_path}'")
    print(f" Best BEN-14K Checkpoint   : '{best_ben_path}'")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Standalone CFM Latent Bridge Trainer")
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
