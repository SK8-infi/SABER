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

from Saber.datasets.ben14k import BEN14KDataset
from Saber.trainer.metrics import compute_retrieval_metrics
import timm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s")
logger = logging.getLogger(__name__)

class ISROModel(nn.Module):
    """
    Reconstructed ISRO Dual-Encoder PVT-v2 Model matching 'best_ben14k_isro_retrieval.pt'.
    """
    def __init__(self):
        super().__init__()
        self.s1_backbone = timm.create_model('pvt_v2_b2', in_chans=6, num_classes=0)
        self.s2_backbone = timm.create_model('pvt_v2_b2', in_chans=16, num_classes=0)
        
        self.s1_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )
        self.s2_projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768)
        )
        self.classifier = nn.Linear(768, 31)
        self.fusion_head = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 31)
        )

    def load_isro_checkpoint(self, path: str, device: torch.device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model_sd = ckpt["model"]
        new_sd = {}
        for k, v in model_sd.items():
            if k.startswith("s1_encoder.backbone."):
                new_sd[k.replace("s1_encoder.backbone.", "s1_backbone.")] = v
            elif k.startswith("s2_encoder.backbone."):
                new_sd[k.replace("s2_encoder.backbone.", "s2_backbone.")] = v
            elif k.startswith("s1_encoder.projection."):
                new_sd[k.replace("s1_encoder.projection.", "s1_projection.")] = v
            elif k.startswith("s2_encoder.projection."):
                new_sd[k.replace("s2_encoder.projection.", "s2_projection.")] = v
            elif k.startswith("classifier."):
                new_sd[k] = v
            elif k.startswith("fusion_head."):
                new_sd[k] = v

        res = self.load_state_dict(new_sd, strict=True)
        logger.info(f"✅ Successfully loaded ISRO Model weights from '{path}' (strict=True)!")
        return ckpt

def prepare_s1_6ch(x_s1: torch.Tensor, s1_mean: torch.Tensor, s1_std: torch.Tensor) -> torch.Tensor:
    """
    Constructs 6-channel SAR input from 2-channel VV/VH.
    Channels: [VV, VH, VV-VH, VV/(VH+eps), VV_norm, VH_norm]
    """
    B, C, H, W = x_s1.shape
    vv = x_s1[:, 0:1, :, :]
    vh = x_s1[:, 1:2, :, :]

    diff = vv - vh
    ratio = vv / (vh.abs() + 1e-4)

    # Normalize VV and VH
    vv_norm = (vv - s1_mean[0]) / (s1_std[0] + 1e-6)
    vh_norm = (vh - s1_mean[1]) / (s1_std[1] + 1e-6)

    return torch.cat([vv, vh, diff, ratio, vv_norm, vh_norm], dim=1)

def prepare_s2_16ch(x_s2: torch.Tensor, s2_mean: torch.Tensor, s2_std: torch.Tensor) -> torch.Tensor:
    """
    Constructs 16-channel Optical input from 12-channel Sentinel-2.
    Channels: [12 S2 Bands Normalized, NDVI, NDWI, NDBI, SAVI]
    """
    B, C, H, W = x_s2.shape
    s2_mean_res = s2_mean.view(1, 12, 1, 1).to(x_s2.device)
    s2_std_res = s2_std.view(1, 12, 1, 1).to(x_s2.device)

    x_s2_norm = (x_s2 - s2_mean_res) / (s2_std_res + 1e-6)

    # Band indices (0-indexed for 12-band Sentinel-2):
    # B2=0, B3=1, B4=2, B8=6, B11=10
    b3 = x_s2[:, 1:2, :, :]
    b4 = x_s2[:, 2:3, :, :]
    b8 = x_s2[:, 6:7, :, :]
    b11 = x_s2[:, 10:11, :, :]

    ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
    ndwi = (b3 - b8) / (b3 + b8 + 1e-6)
    ndbi = (b11 - b8) / (b11 + b8 + 1e-6)
    savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5 + 1e-6)

    return torch.cat([x_s2_norm, ndvi, ndwi, ndbi, savi], dim=1)

def evaluate_isro_model(
    checkpoint_path: str = "best_ben14k_isro_retrieval.pt",
    data_dir: str = "Datasets/benv1_14k",
    batch_size: int = 64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 80)
    logger.info(" 🔍 ISRO MODEL EVALUATION & BENCHMARK RUNNER")
    logger.info("=" * 80)
    logger.info(f" Device: {device} | Checkpoint: '{checkpoint_path}' | Data Dir: '{data_dir}'")

    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint file '{checkpoint_path}' not found!")
        return

    # Load Model
    model = ISROModel().to(device)
    ckpt_meta = model.load_isro_checkpoint(checkpoint_path, device)
    model.eval()

    cfg = ckpt_meta.get("config", {})
    s1_mean = torch.tensor(cfg.get("s1_mean", [0.349, 0.4176]), device=device)
    s1_std = torch.tensor(cfg.get("s1_std", [0.1345, 0.1247]), device=device)
    s2_mean = torch.tensor(cfg.get("s2_mean", [0.0348] * 12), device=device)
    s2_std = torch.tensor(cfg.get("s2_std", [0.0192] * 12), device=device)

    # Load Test Dataset
    logger.info(f"Loading BEN-14K [TEST SPLIT] from '{data_dir}'...")
    test_dataset = BEN14KDataset(data_dir=data_dir, modality="both", split="test", is_train=False, use_synthetic=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("⚡ Extracting S1 and S2 Embeddings & Semantic Logits for Test Set...")
    z1_list, z2_list, p1_list, p2_list, labels_list, names_list = [], [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Test Features", dynamic_ncols=True):
            images = batch.get("image", batch.get("image1")).to(device, non_blocking=True)
            if images.ndim == 4 and (images.shape[-1] != 224 or images.shape[-2] != 224):
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

            raw_s1 = images[:, :2, :, :]
            raw_s2 = images[:, 2:, :, :]

            x1_6ch = prepare_s1_6ch(raw_s1, s1_mean, s1_std)
            x2_16ch = prepare_s2_16ch(raw_s2, s2_mean, s2_std)

            f1 = model.s1_backbone(x1_6ch)
            f2 = model.s2_backbone(x2_16ch)

            z1 = F.normalize(model.s1_projection(f1), p=2, dim=-1)
            z2 = F.normalize(model.s2_projection(f2), p=2, dim=-1)

            p1 = torch.sigmoid(model.classifier(z1))
            p2 = torch.sigmoid(model.classifier(z2))

            z1_list.append(z1.cpu())
            z2_list.append(z2.cpu())
            p1_list.append(p1.cpu())
            p2_list.append(p2.cpu())
            labels_list.append(batch["label"].cpu())
            names_list.extend(batch["name"])

    all_z1 = torch.cat(z1_list, dim=0).numpy()
    all_z2 = torch.cat(z2_list, dim=0).numpy()
    all_p1 = torch.cat(p1_list, dim=0).numpy()
    all_p2 = torch.cat(p2_list, dim=0).numpy()
    all_labels = torch.cat(labels_list, dim=0).numpy()
    all_names = np.array(names_list)

    N_samples = all_z1.shape[0]

    # Standard query (20%) / gallery (80%) split with fixed seed 42
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(N_samples)
    q_size = max(1, N_samples // 5)
    q_idx = np.sort(shuffled[:q_size])
    g_idx = np.sort(shuffled[q_size:])

    logger.info(f"📊 Test Split: {len(q_idx)} Query Items -> {len(g_idx)} Gallery Items.")

    # Descriptor Configurations
    spec = ckpt_meta.get("best_descriptor_spec", {"embedding_weight": 0.4, "semantic_weight": 0.6})
    w_emb = spec.get("embedding_weight", 0.4)
    w_sem = spec.get("semantic_weight", 0.6)

    # Compute hybrid descriptors
    h1 = F.normalize(torch.tensor(w_emb * all_z1 + w_sem * all_p1), p=2, dim=-1).numpy()
    h2 = F.normalize(torch.tensor(w_emb * all_z2 + w_sem * all_p2), p=2, dim=-1).numpy()

    eval_modes = [
        ("Same-Modal (S2 -> S2) [Pure Latents]", all_z2[q_idx], all_z2[g_idx]),
        ("Same-Modal (S2 -> S2) [Hybrid 0.4z+0.6p]", h2[q_idx], h2[g_idx]),
        ("Cross-Modal (S1 -> S2) [Pure Latents]", all_z1[q_idx], all_z2[g_idx]),
        ("Cross-Modal (S1 -> S2) [Hybrid 0.4z+0.6p]", h1[q_idx], h2[g_idx]),
        ("Same-Modal (S1 -> S1) [Pure Latents]", all_z1[q_idx], all_z1[g_idx]),
        ("Same-Modal (S1 -> S1) [Hybrid 0.4z+0.6p]", h1[q_idx], h1[g_idx]),
        ("Cross-Modal (S2 -> S1) [Pure Latents]", all_z2[q_idx], all_z1[g_idx]),
        ("Cross-Modal (S2 -> S1) [Hybrid 0.4z+0.6p]", h2[q_idx], h1[g_idx]),
    ]

    print("\n" + "=" * 90)
    print(" 🏆 ISRO MODEL BENCHMARK RETRIEVAL RESULTS (BEN-14K HELD-OUT TEST SET)")
    print("=" * 90)
    print(f"{'Retrieval Direction & Strategy':<45} | {'MAP@5':<8} | {'F1@5':<8} | {'Prec@5':<8} | {'Rec@5':<8}")
    print("-" * 90)

    for title, q_emb, g_emb in eval_modes:
        res = compute_retrieval_metrics(
            query_embeds=q_emb,
            gallery_embeds=g_emb,
            query_labels=all_labels[q_idx],
            gallery_labels=all_labels[g_idx],
            top_k=5,
            is_multilabel=True,
            query_names=all_names[q_idx],
            gallery_names=all_names[g_idx],
            exclude_self_matches=False
        )
        map5 = res.get("map@5", res.get("MAP@5", 0.0))
        f1_5 = res.get("f1@5", res.get("F1@5", 0.0))
        prec5 = res.get("precision@5", res.get("PRECISION@5", 0.0))
        rec5 = res.get("recall@5", res.get("RECALL@5", 0.0))

        print(f"{title:<45} | {map5:<8.4f} | {f1_5:<8.4f} | {prec5:<8.4f} | {rec5:<8.4f}")

    print("=" * 90 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ISRO Retrieval Model on BEN-14K")
    parser.add_argument("--checkpoint", type=str, default="best_ben14k_isro_retrieval.pt", help="Path to ISRO checkpoint")
    parser.add_argument("--data_dir", type=str, default="Datasets/benv1_14k", help="Path to BEN-14K dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    evaluate_isro_model(checkpoint_path=args.checkpoint, data_dir=args.data_dir, batch_size=args.batch_size)
