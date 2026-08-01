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

    print(f"=== Running SABER Comprehensive EDA Diagnostics on '{resolved_path}' ===")
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
    model = SABER(config=config, in_channels=dataset.num_channels).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
        state_dict = ckpt["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.")}
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model weights.")
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running with initial weights.")
        
    model.eval()
    
    # Extract Embeddings
    print("Extracting embeddings for evaluation dataset...")
    embeds_list, labels_list, names_list = [], [], []
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            if images.shape[-1] != 224 or images.shape[-2] != 224:
                import torch.nn.functional as F
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            embeds = model.get_retrieval_embedding(images)
            embeds_list.append(embeds.cpu().numpy())
            labels_list.append(batch["label"].numpy())
            names_list.extend(batch["name"])
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"Extraction progress: Batch [{batch_idx + 1}/{num_batches}] ({((batch_idx + 1)/num_batches)*100:.1f}%)", flush=True)
            
    embeddings = np.concatenate(embeds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    names = np.array(names_list)
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
        # Queries that have class c_idx active
        c_query_mask = (q_lbl_t[:, c_idx] > 0)
        c_count = c_query_mask.sum().item()
        
        if c_count == 0:
            continue
            
        # Top-5 retrieved gallery items for these queries
        c_retrieved = top5_g_labels[c_query_mask, :, c_idx] # (Q_c, 5)
        
        # Hits per rank
        hits = (c_retrieved > 0).float()
        
        # Precision@5 for class: hits / 5
        c_precision = hits.mean().item()
        
        # Recall@5 for class: hits / total active labels in query
        c_active_total = q_lbl_t[c_query_mask].sum(dim=1, keepdim=True) # (Q_c, 1)
        c_recall = (hits.sum(dim=1, keepdim=True) / (c_active_total + 1e-8)).mean().item()
        
        c_f1 = (2 * c_precision * c_recall) / (c_precision + c_recall + 1e-8)
        
        class_results.append({
            "class": class_name,
            "count": c_count,
            "precision": c_precision,
            "recall": c_recall,
            "f1": c_f1
        })
        
    class_results.sort(key=lambda x: x["f1"])
    
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
    
    # Compute Jaccard overlap matrix for all Q x G
    intersection = torch.matmul(q_lbl_t, g_lbl_t.t()) # (Q, G)
    sum_q = q_lbl_t.sum(dim=1, keepdim=True)
    sum_g = g_lbl_t.sum(dim=1, keepdim=True)
    union = sum_q + sum_g.t() - intersection
    jaccard = intersection / (union + 1e-8)
    jaccard.masked_fill_(~mask, 0.0)
    
    # For each query, get sorted Jaccard values along predicted ranks
    ranked_jaccard = torch.gather(jaccard, 1, sorted_indices) # (Q, G)
    
    first_hit_ranks = []
    top1_count, top5_count, top10_count, top20_count, miss_count = 0, 0, 0, 0, 0
    
    for q_i in range(len(q_embeds)):
        # Find first rank where Jaccard >= 0.25
        match_ranks = torch.where(ranked_jaccard[q_i] >= 0.25)[0]
        if len(match_ranks) > 0:
            first_rank = match_ranks[0].item() + 1 # 1-indexed rank
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
    print(f"Top-1 First Match Rate  : {top1_count/total_q*100:.2f}% ({top1_count}/{total_q})")
    print(f"Top-5 First Match Rate  : {top5_count/total_q*100:.2f}% ({top5_count}/{total_q})")
    print(f"Top-10 First Match Rate : {top10_count/total_q*100:.2f}% ({top10_count}/{total_q})")
    print(f"Top-20 First Match Rate : {top20_count/total_q*100:.2f}% ({top20_count}/{total_q})")
    print(f"Near-Misses (Ranks 6-15): {(top10_count - top5_count)/total_q*100:.2f}% of queries sit just outside Top-5 cutoff!")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: Label Confusion & Misretrieval Co-occurrence
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 🔀 DIAGNOSTIC 3: TOP MISRETRIEVED CLASS CONFUSION PAIRS")
    print("="*80)
    
    confusion_matrix = torch.zeros((19, 19), device=device)
    for c_i in range(19):
        c_queries = (q_lbl_t[:, c_i] > 0)
        if c_queries.sum() > 0:
            retrieved_top5 = top5_g_labels[c_queries].sum(dim=1) # sum over top-5
            confusion_matrix[c_i] = retrieved_top5.sum(dim=0)
            
    # Normalize rows
    row_sums = confusion_matrix.sum(dim=1, keepdim=True) + 1e-8
    norm_conf = confusion_matrix / row_sums
    
    pairs = []
    for i in range(19):
        for j in range(19):
            if i != j and norm_conf[i, j].item() > 0.05:
                pairs.append((BIGEARTHNET_19_CLASSES[i], BIGEARTHNET_19_CLASSES[j], norm_conf[i, j].item()))
                
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"{'Query Class':<35} -> {'Incorrectly Retrieved Class':<35} | {'Confusion Frequency'}")
    print("-" * 88)
    for q_c, r_c, freq in pairs[:12]:
        print(f"{q_c:<35} -> {r_c:<35} | {freq*100:.2f}%")

    # -------------------------------------------------------------
    # DIAGNOSTIC 4: Embedding Geometry & Anisotropy (SVD) Audit
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 📐 DIAGNOSTIC 4: EMBEDDING SPACE GEOMETRY & ANISOTROPY (SVD)")
    print("="*80)
    
    # SVD of gallery embeddings
    g_centered = g_embeds - np.mean(g_embeds, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(g_centered, full_matrices=False)
    
    var_explained = (S ** 2) / np.sum(S ** 2)
    cum_var = np.cumsum(var_explained)
    
    print(f"Top 5 Singular Values account for  : {cum_var[4]*100:.2f}% of total feature variance")
    print(f"Top 10 Singular Values account for : {cum_var[9]*100:.2f}% of total feature variance")
    print(f"Top 50 Singular Values account for : {cum_var[49]*100:.2f}% of total feature variance")
    
    if cum_var[9] > 0.70:
        print("WARNING: ANISOTROPY DETECTED! Features occupy a low-rank subspace.")
    else:
        print("OK: Feature space utilizes high-dimensional variance effectively.")
        
    # Cosine Similarity Distribution
    mean_sim_all = torch.mean(sims[mask]).item()
    print(f"Mean Global Cosine Similarity (Random Pairs) : {mean_sim_all:.4f}")
    
    pos_mask = (jaccard > 0.5) & mask
    if pos_mask.any():
        mean_sim_pos = torch.mean(sims[pos_mask]).item()
        print(f"Mean Matching Cosine Similarity (Jaccard > 0.5): {mean_sim_pos:.4f}")
        print(f"Cosine Dynamic Range Gap                     : {mean_sim_pos - mean_sim_all:.4f}")
    
    # -------------------------------------------------------------
    # DIAGNOSTIC 5: Visual Failure Case Summary ("Hall of Shame")
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print(" 🖼️ DIAGNOSTIC 5: WORST PERFORMING QUERY EXAMPLES (0% F1@5)")
    print("="*80)
    
    # Calculate per-query F1@5
    q_prec_5 = (top5_g_labels * q_lbl_t.unsqueeze(1)).sum(dim=2) / (top5_g_labels.sum(dim=2) + 1e-8)
    q_rec_5 = (top5_g_labels * q_lbl_t.unsqueeze(1)).sum(dim=2) / (q_lbl_t.sum(dim=1, keepdim=True) + 1e-8)
    q_f1_5 = ((2 * q_prec_5 * q_rec_5) / (q_prec_5 + q_rec_5 + 1e-8)).mean(dim=1)
    
    zero_f1_indices = torch.where(q_f1_5 == 0)[0].cpu().numpy()
    print(f"Total Queries with 0.00% F1@5 score: {len(zero_f1_indices)} / {total_q} ({len(zero_f1_indices)/total_q*100:.2f}%)")
    
    print("\nSample Worst 5 Queries:")
    for idx in zero_f1_indices[:5]:
        q_name = q_names[idx]
        q_active_classes = [BIGEARTHNET_19_CLASSES[c] for c in range(19) if q_labels[idx, c] > 0]
        top1_retrieved_name = g_names[sorted_indices[idx, 0].item()]
        top1_retrieved_classes = [BIGEARTHNET_19_CLASSES[c] for c in range(19) if g_labels[sorted_indices[idx, 0].item(), c] > 0]
        
        print(f"\nQuery Name       : {q_name}")
        print(f"True Labels      : {', '.join(q_active_classes)}")
        print(f"Top-1 Retrieved  : {top1_retrieved_name}")
        print(f"Retrieved Labels : {', '.join(top1_retrieved_classes)}")

    print("\n" + "="*80)
    print(" === EDA DIAGNOSTICS COMPLETE ===")
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
