import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.ben14k import BEN14KDataset, BIGEARTHNET_19_CLASSES
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER

def resolve_existing_path(path: str, candidate_paths: list) -> str:
    if path and os.path.exists(path):
        return path
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return path or ""

def run_eda_diagnostics(
    checkpoint_path: str = "checkpoints/saber_unified.pth",
    config_path: str = "Saber/configs/config.yaml",
    data_dir_override: str = None
):
    candidate_paths = [
        checkpoint_path,
        "checkpoints/saber_unified.pth",
        "checkpoints/latest.pth",
        "checkpoints/latest_ben14k.pth",
        "/content/drive/MyDrive/SABER_Data/checkpoints/saber_unified.pth"
    ]
    resolved_path = resolve_existing_path(checkpoint_path, candidate_paths)

    print(f"=== Running SABER Advanced EDA Diagnostics on '{resolved_path}' ===")
    checkpoint_path = resolved_path
    config = load_config(config_path)
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"Computation Device: {device}")
    
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

    # Load dataset
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)
    dataset = BEN14KDataset(
        data_dir=ben_resolved_path,
        use_synthetic=config.dataset.use_synthetic,
        size=config.dataset.get("size", 14832),
        image_size=config.dataset.image_size,
        transform=eval_transform,
        modality=config.dataset.get("modality", "s2"),
        is_train=False,
        split="all"
    )
    
    num_workers = config.dataset.get("num_workers", 2)
    batch_size = 128 if torch.cuda.is_available() else 32
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Load SABER model
    model = SABER(config=config, in_channels=14).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.") and not k.startswith("classifier.")}
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model weights.")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running with initialized weights.")
        
    model.eval()
    
    # Extract features
    print("Extracting embeddings for evaluation dataset...")
    embeds_list = []
    labels_list = []
    names_list = []
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch.get("image1", batch.get("image")).to(device)
            if imgs.shape[-1] != 224 or imgs.shape[-2] != 224:
                imgs = torch.nn.functional.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)
            
            wvs = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.610, 2.190] if imgs.shape[1] >= 12 else model.s2_wvs
            if imgs.shape[1] > len(wvs):
                wvs = model.s1_wvs + model.s2_wvs
                
            feats = model.backbone(imgs, wvs)
            embeds = model.projection_head(feats)
            embeds = torch.nn.functional.normalize(embeds, dim=-1)
            
            embeds_list.append(embeds.cpu().numpy())
            labels_list.append(batch["label"].numpy())
            names_list.append(batch["name"])
            
    embeddings = np.concatenate(embeds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    names = np.concatenate(names_list, axis=0)
    
    num_samples = len(embeddings)
    print(f"Extracted {num_samples} samples. Embedding dim: {embeddings.shape[1]}")
    
    # Random split: 20% query, 80% gallery
    rng = np.random.RandomState(42)
    shuffled_indices = rng.permutation(num_samples)
    query_size = max(1, num_samples // 5)
    query_idx = np.sort(shuffled_indices[:query_size])
    gallery_idx = np.arange(num_samples)
    
    q_embeds = embeddings[query_idx]
    q_labels = labels[query_idx]
    q_names = names[query_idx]
    
    g_embeds = embeddings[gallery_idx]
    g_labels = labels[gallery_idx]
    g_names = names[gallery_idx]
    
    # Compute similarity matrix (Q x G)
    print("Computing cosine similarity matrix...")
    q_tensor = torch.tensor(q_embeds, device=device)
    g_tensor = torch.tensor(g_embeds, device=device)
    sims = torch.matmul(q_tensor, g_tensor.t())
    
    # Mask out self-matches
    mask_cpu = q_names[:, None] != g_names[None, :]
    mask = torch.tensor(mask_cpu, device=device)
    sims = sims.masked_fill(~mask, float('-inf'))
    
    # Sorted indices per query
    sorted_sims, sorted_indices = torch.sort(sims, dim=1, descending=True)
    
    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Per-Class Precision, Recall, F1 Breakdown
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 📊 DIAGNOSTIC 1: PER-CLASS ACCURACY & BREAKDOWN (Top-5)")
    print("="*80)
    
    q_lbl_t = torch.tensor(q_labels, dtype=torch.float32, device=device)
    g_lbl_t = torch.tensor(g_labels, dtype=torch.float32, device=device)
    
    top5_indices = sorted_indices[:, :5] # (Q, 5)
    top5_g_labels = g_lbl_t[top5_indices] # (Q, 5, C)
    
    # Compute class-wise metrics
    class_results = []
    for c_idx, class_name in enumerate(BIGEARTHNET_19_CLASSES):
        c_query_mask = (q_lbl_t[:, c_idx] > 0)
        c_count = c_query_mask.sum().item()
        
        if c_count == 0:
            continue
            
        c_retrieved = top5_g_labels[c_query_mask, :, c_idx] # (Q_c, 5)
        hits = (c_retrieved > 0).float()
        
        c_precision = hits.mean().item()
        c_active_total = q_lbl_t[c_query_mask].sum(dim=1, keepdim=True)
        c_recall = (hits.sum(dim=1, keepdim=True) / (c_active_total + 1e-8)).mean().item()
        c_f1 = (2 * c_precision * c_recall) / (c_precision + c_recall + 1e-8)
        
        class_results.append({
            "class": class_name,
            "count": c_count,
            "precision": c_precision,
            "recall": c_recall,
            "f1": c_f1
        })
        
    class_results.sort(key=lambda x: x['f1'], reverse=True)
    
    print(f"{'Class Name':<42} | {'Count':<6} | {'Precision@5':<12} | {'Recall@5':<10} | {'F1@5':<8}")
    print("-" * 88)
    for res in class_results:
        status_flag = "[DEAD]" if res['f1'] < 0.25 else ("[HIGH]" if res['f1'] > 0.65 else " [OK] ")
        print(f"{res['class']:<42} | {res['count']:<6} | {res['precision']*100:<11.2f}% | {res['recall']*100:<9.2f}% | {res['f1']*100:<6.2f}% {status_flag}")
        
    # -------------------------------------------------------------
    # DIAGNOSTIC 2: Rank Distribution & Near-Miss Analysis
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 🎯 DIAGNOSTIC 2: RANK DISTRIBUTION & NEAR-MISS ANALYSIS")
    print("="*80)
    
    intersection = torch.matmul(q_lbl_t, g_lbl_t.t()) # (Q, G)
    sum_q = q_lbl_t.sum(dim=1, keepdim=True)
    sum_g = g_lbl_t.sum(dim=1, keepdim=True)
    union = sum_q + sum_g.t() - intersection
    jaccard = intersection / (union + 1e-8)
    jaccard.masked_fill_(~mask, 0.0)
    
    ranked_jaccard = torch.gather(jaccard, 1, sorted_indices)
    
    first_hit_ranks = []
    top1_count, top5_count, top10_count, top20_count, miss_count = 0, 0, 0, 0, 0
    
    for q_i in range(len(q_embeds)):
        match_ranks = torch.where(ranked_jaccard[q_i] >= 0.25)[0]
        if len(match_ranks) > 0:
            first_rank = match_ranks[0].item() + 1
            first_hit_ranks.append(first_rank)
            if first_rank == 1:
                top1_count += 1
            if first_rank <= 5:
                top5_count += 1
            if first_rank <= 10:
                top10_count += 1
            if first_rank <= 20:
                top20_count += 1
        else:
            miss_count += 1
            
    total_q = len(q_embeds)
    print(f"Top-1 Hit Rate   : {top1_count/total_q*100:.2f}% ({top1_count}/{total_q})")
    print(f"Top-5 Hit Rate   : {top5_count/total_q*100:.2f}% ({top5_count}/{total_q})")
    print(f"Top-10 Hit Rate  : {top10_count/total_q*100:.2f}% ({top10_count}/{total_q})")
    print(f"Top-20 Hit Rate  : {top20_count/total_q*100:.2f}% ({top20_count}/{total_q})")
    print(f"Complete Misses  : {miss_count/total_q*100:.2f}% ({miss_count}/{total_q})")
    if first_hit_ranks:
        print(f"Median First Hit Rank: {np.median(first_hit_ranks):.1f}")
        
    # -------------------------------------------------------------
    # DIAGNOSTIC 3: Anisotropy & Cosine Distance Spread
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 📐 DIAGNOSTIC 3: FEATURE SPACE ANISOTROPY & DYNAMIC RANGE")
    print("="*80)
    
    mean_sim_all = torch.mean(sims[mask]).item()
    print(f"Mean Global Cosine Similarity (Random Pairs) : {mean_sim_all:.4f}")
    
    pos_mask = (jaccard > 0.5) & mask
    if pos_mask.any():
        mean_sim_pos = torch.mean(sims[pos_mask]).item()
        print(f"Mean Matching Cosine Similarity (Jaccard > 0.5): {mean_sim_pos:.4f}")
        print(f"Cosine Dynamic Range Gap                     : {mean_sim_pos - mean_sim_all:.4f}")
        
    # -------------------------------------------------------------
    # DIAGNOSTIC 4: 19x19 LABEL CO-OCCURRENCE ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 🔗 DIAGNOSTIC 4: TOP CO-OCCURRING LABEL PAIRS (CONDITIONAL DEPENDENCIES)")
    print("="*80)
    
    lbl_matrix = torch.tensor(labels, dtype=torch.float32, device=device) # (N, 19)
    co_occur = torch.matmul(lbl_matrix.t(), lbl_matrix) # (19, 19)
    class_counts = lbl_matrix.sum(dim=0) # (19,)
    
    # Conditional probability P(j | i) = co_occur[i, j] / count[i]
    cond_prob = co_occur / (class_counts.unsqueeze(1) + 1e-8)
    
    co_pairs = []
    for i in range(19):
        for j in range(19):
            if i != j and class_counts[i] > 0:
                co_pairs.append((
                    BIGEARTHNET_19_CLASSES[i],
                    BIGEARTHNET_19_CLASSES[j],
                    co_occur[i, j].item(),
                    cond_prob[i, j].item()
                ))
    co_pairs.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'Source Class (Query)':<35} -> {'Co-Occurring Target':<35} | {'P(Target|Source)':<18}")
    print("-" * 92)
    for p in co_pairs[:8]:
        print(f"{p[0]:<35} -> {p[1]:<35} | {p[3]*100:<17.2f}%")

    # -------------------------------------------------------------
    # DIAGNOSTIC 5: PERFORMANCE BREAKDOWN BY ACTIVE LABEL DENSITY
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 🏷️ DIAGNOSTIC 5: F1-SCORE BREAKDOWN BY ACTIVE LABEL DENSITY PER TILE")
    print("="*80)
    
    q_active_counts = q_lbl_t.sum(dim=1).cpu().numpy()
    q_prec_5 = (top5_g_labels * q_lbl_t.unsqueeze(1)).sum(dim=2) / (top5_g_labels.sum(dim=2) + 1e-8)
    q_rec_5 = (top5_g_labels * q_lbl_t.unsqueeze(1)).sum(dim=2) / (q_lbl_t.sum(dim=1, keepdim=True) + 1e-8)
    q_f1_5 = ((2 * q_prec_5 * q_rec_5) / (q_prec_5 + q_rec_5 + 1e-8)).mean(dim=1).cpu().numpy()
    
    print(f"{'Active Label Count':<22} | {'Query Count':<12} | {'Avg F1@5':<12}")
    print("-" * 52)
    for count_val in range(1, 8):
        density_mask = (q_active_counts == count_val)
        n_queries = density_mask.sum()
        if n_queries > 0:
            avg_f1_density = q_f1_5[density_mask].mean()
            print(f"{count_val:<22} | {n_queries:<12} | {avg_f1_density*100:<11.2f}%")
            
    print("\n" + "="*80)
    print(" === ADVANCED EDA DIAGNOSTICS COMPLETE ===")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SABER Comprehensive EDA Diagnostics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/saber_unified.pth", help="Path to SABER model checkpoint")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to BEN-14K dataset directory")
    args = parser.parse_args()

    run_eda_diagnostics(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_dir_override=args.data_dir
    )
