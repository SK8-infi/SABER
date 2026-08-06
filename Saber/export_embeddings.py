import sys
import os
import time
import argparse
import logging
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.logger import setup_logger
from Saber.datasets.ben14k import BEN14KDataset
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

def resolve_existing_path(path: str, candidate_paths: list) -> str:
    if path and os.path.exists(path):
        return path
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return path or ""

def export_embeddings(
    config_path: str = "Saber/configs/config.yaml",
    checkpoint_path: str = "checkpoints_v10/saber_unified_clean.pth",
    bridge_path: str = "checkpoints_v10/bridge_best_ben14k.pth",
    output_db_path: str = "saber_search_db.pth",
    batch_size: int = 64
):
    logger = setup_logger("saber_export")
    logger.info("=" * 80)
    logger.info(" 🚀 SABER EMBEDDING & FAISS DATABASE EXPORTER")
    logger.info("=" * 80)

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation Device: {device}")

    # Resolve dataset path
    ben_raw_path = config.dataset.data_dir
    ben_resolved_path = resolve_existing_path(
        ben_raw_path,
        [
            "Datasets/benv1_14k",
            "datasets/benv1_14k",
            "/content/SABER/Datasets/benv1_14k",
            "/content/Datasets/benv1_14k"
        ]
    )

    logger.info(f"Loading full BEN-14K dataset from '{ben_resolved_path}'...")
    dataset = BEN14KDataset(
        data_dir=ben_resolved_path,
        use_synthetic=False,
        size=14832,
        image_size=224,
        modality="both",
        is_train=False,
        split="all"  # Export full dataset
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    N_samples = len(dataset)
    logger.info(f"Dataset Loaded: {N_samples} total multi-modal samples.")

    # Load Model
    logger.info("Instantiating SABER model...")
    model = SABER(config=config, in_channels=14).to(device)

    ckpt_resolved = resolve_existing_path(
        checkpoint_path,
        [
            "checkpoints_v10/saber_unified_clean.pth",
            "checkpoints_v10/40epochs/saber_unified.pth",
            "checkpoints/saber_unified_clean.pth"
        ]
    )
    if os.path.exists(ckpt_resolved):
        logger.info(f"Loading master encoder checkpoint: '{ckpt_resolved}'")
        try:
            ckpt = torch.load(ckpt_resolved, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_resolved, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    else:
        logger.warning(f"⚠️ Checkpoint not found at '{ckpt_resolved}'. Using initial model weights.")

    # Load CFM Bridge
    bridge_resolved = resolve_existing_path(
        bridge_path,
        [
            "checkpoints_v10/bridge_best_ben14k.pth",
            "checkpoints_v10/40epochs/bridge_unified.pth",
            "checkpoints/bridge_best.pth"
        ]
    )
    bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=4, dropout=0.1).to(device)
    if os.path.exists(bridge_resolved):
        logger.info(f"Loading CFM Bridge checkpoint: '{bridge_resolved}'")
        try:
            b_ckpt = torch.load(bridge_resolved, map_location=device, weights_only=False)
        except TypeError:
            b_ckpt = torch.load(bridge_resolved, map_location=device)
        b_sd = b_ckpt.get("bridge_state_dict", b_ckpt.get("state_dict", b_ckpt))
        bridge_net.load_state_dict(b_sd, strict=False)
    
    model.bridge.cfm_bridge = bridge_net
    model.bridge.ode_steps = 10
    model.eval()

    # Pre-extract all features
    logger.info("⚡ Extracting S1, S2, Translated S1->S2 vectors and Semantic Class Logits...")
    s1_list, s2_list, s1_trans_list, p1_list, p2_list, labels_list, names_list = [], [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Database Embeddings", dynamic_ncols=True):
            images = batch.get("image", batch.get("image1")).to(device, non_blocking=True)
            if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

            x_s1 = images[:, :2, :, :]
            x_s2 = images[:, 2:, :, :]

            z1_raw, z2_raw, _, logits_s1, logits_s2 = model(x_s1, x_s2)
            z1 = F.normalize(z1_raw, p=2, dim=-1)
            z2 = F.normalize(z2_raw, p=2, dim=-1)

            p1 = torch.sigmoid(logits_s1)
            p2 = torch.sigmoid(logits_s2)

            # CFM Bridge translation S1 -> S2
            z1_trans_raw = model.bridge(z1, c_class=p1)
            z1_trans = F.normalize(z1_trans_raw, p=2, dim=-1)

            s1_list.append(z1.cpu().numpy())
            s2_list.append(z2.cpu().numpy())
            s1_trans_list.append(z1_trans.cpu().numpy())
            p1_list.append(p1.cpu().numpy())
            p2_list.append(p2.cpu().numpy())
            labels_list.append(batch["label"].cpu().numpy())
            names_list.extend(batch["name"])

    all_s1 = np.concatenate(s1_list, axis=0)
    all_s2 = np.concatenate(s2_list, axis=0)
    all_s1_trans = np.concatenate(s1_trans_list, axis=0)
    all_p1 = np.concatenate(p1_list, axis=0)
    all_p2 = np.concatenate(p2_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_names = np.array(names_list)

    # High-F1 Class Probability Thresholding (0.15 Noise Removal)
    p1_t = np.where(all_p1 > 0.15, all_p1, 0.0)
    p2_t = np.where(all_p2 > 0.15, all_p2, 0.0)

    p1_norm = p1_t / (np.linalg.norm(p1_t, axis=1, keepdims=True) + 1e-8)
    p2_norm = p2_t / (np.linalg.norm(p2_t, axis=1, keepdims=True) + 1e-8)

    # 787-D High-F1 Multi-Label Hybrid Descriptors (0.6 Visual Vector + 0.4 Thresholded Class Vector)
    v1_c = np.hstack([0.6 * all_s1_trans, 0.4 * p1_norm])
    v2_c = np.hstack([0.6 * all_s2, 0.4 * p2_norm])

    h1 = (v1_c / (np.linalg.norm(v1_c, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    h2 = (v2_c / (np.linalg.norm(v2_c, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    # Apply Database Augmentation (DBA) Gallery Manifold Smoothing
    logger.info("Applying Database Augmentation (DBA) gallery manifold smoothing...")
    g_norm = h2 / (np.linalg.norm(h2, axis=1, keepdims=True) + 1e-8)
    sims = g_norm @ g_norm.T
    np.fill_diagonal(sims, -1.0)
    top_k_idx = np.argpartition(sims, -3, axis=1)[:, -3:]
    top_k_weights = np.take_along_axis(sims, top_k_idx, axis=1)[:, :, None]
    h2_dba = g_norm + 1.5 * np.sum(g_norm[top_k_idx] * top_k_weights, axis=1)
    h2_dba = h2_dba / (np.linalg.norm(h2_dba, axis=1, keepdims=True) + 1e-8)

    # Build FAISS Indices on DBA-Smoothed Gallery Vectors if available
    faiss_s2_index = None
    if FAISS_AVAILABLE:
        logger.info("Building FAISS Flat Cosine Index for DBA-Smoothed Optical Gallery (787-D)...")
        index = faiss.IndexFlatIP(h2_dba.shape[1])
        index.add(h2_dba.astype(np.float32))
        faiss_s2_index = faiss.serialize_index(index)
        logger.info("✅ FAISS S2 DBA-Smoothed Index successfully built!")

    # Package Export Database
    db_payload = {
        "num_samples": N_samples,
        "names": all_names,
        "labels": all_labels,
        "s1_embeds": all_s1.astype(np.float16),           # Half precision to save 50% disk space
        "s2_embeds": all_s2.astype(np.float16),
        "s1_translated_embeds": all_s1_trans.astype(np.float16),
        "class_probs_s1": all_p1.astype(np.float16),
        "class_probs_s2": all_p2.astype(np.float16),
        "hybrid_s1_embeds": h1.astype(np.float16),
        "hybrid_s2_embeds": h2.astype(np.float16),
        "faiss_s2_index": faiss_s2_index,
        "class_names": getattr(config.dataset, "class_names", [
            "Urban fabric", "Industrial units", "Arable land", "Permanent crops",
            "Pastures", "Complex agriculture", "Forests", "Shrub/herbaceous",
            "Open spaces", "Inland wetlands", "Coastal wetlands", "Inland waters",
            "Marine waters", "Water bodies", "Natural grassland", "Moors and heathland",
            "Sparsely vegetated", "Burnt areas", "Bare rock"
        ])
    }

    torch.save(db_payload, output_db_path)
    db_size_mb = os.path.getsize(output_db_path) / (1024 * 1024)

    logger.info("=" * 80)
    logger.info(f" 🎉 SUCCESS! Database saved to '{output_db_path}' ({db_size_mb:.2f} MB)")
    logger.info(" Download this file to your laptop and run 'python local_search_engine.py'!")
    logger.info("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SABER Embeddings & FAISS Database")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v10/saber_unified_clean.pth")
    parser.add_argument("--bridge", type=str, default="checkpoints_v10/bridge_best_ben14k.pth")
    parser.add_argument("--output", type=str, default="saber_search_db.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    export_embeddings(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        bridge_path=args.bridge,
        output_db_path=args.output,
        batch_size=args.batch_size
    )
