import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from Saber.utils.config import load_config
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.ben14k import BEN14KDataset
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import FAISSIndex
from Saber.retrieval.retriever import Retriever
from Saber.retrieval.rerank import ReciprocalReranker

def calculate_jaccard(labels1: np.ndarray, labels2: np.ndarray) -> float:
    b1 = labels1 > 0.5
    b2 = labels2 > 0.5
    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)

def calculate_recall_class_coverage(q_label: np.ndarray, top_k_labels: list) -> float:
    """Calculates recall as fraction of query's active classes covered by retrieved top-K items."""
    q_classes = set(np.where(q_label > 0.5)[0])
    if not q_classes:
        return 1.0
    retrieved_classes = set()
    for lbl in top_k_labels:
        retrieved_classes.update(np.where(lbl > 0.5)[0])
    intersection = q_classes.intersection(retrieved_classes)
    return len(intersection) / float(len(q_classes))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== COMPREHENSIVE EMPIRICAL ABLATION EXPERIMENT (500 QUERIES on {device}) ===")

    # Load 500 test query scenes
    num_queries = 500
    ds = BEN14KDataset(data_dir="datasets/benv1_14k", use_synthetic=False, is_train=False, split="all", modality="both", size=2000)

    # Load Gallery Metadata & Embeddings
    meta_saber = torch.load("checkpoints/crossmodal/faiss_index_metadata.pth", map_location="cpu", weights_only=False)
    gallery_names = meta_saber["names"]
    gallery_labels = meta_saber["labels"]
    gallery_embs_saber = meta_saber["embeddings"]

    idx_saber = FAISSIndex(dimension=768)
    idx_saber.build_index(gallery_embs_saber)

    # Load SABER Model
    cfg = load_config("Saber/configs/config.yaml")
    saber = SABER(config=cfg, in_channels=14).to(device)
    ckpt_saber = load_checkpoint("checkpoints/latest_ben14k.pth", map_location=str(device))
    saber.load_state_dict(ckpt_saber["model_state_dict"], strict=False)
    saber.eval()

    # Pre-extract query features for ode_steps ∈ {1, 2, 3, 4, 5, 8, 10}
    ode_steps_list = [1, 2, 3, 4, 5, 8, 10]
    shortlist_k_list = [10, 15, 20, 25, 30, 40, 50, 75, 100]

    print("Pre-computing query feature embeddings across ODE steps...")
    query_data = [] # list of dicts: {'label': np, 'embs': {ode_step: np_emb}, 'lat_backbone': float, 'lat_ode': {ode_step: float}}

    for q_i in tqdm(range(num_queries), desc="Extracting Query Embeddings"):
        sample = ds[q_i]
        q_label = sample["label"].numpy()
        img_s1 = sample.get("image_s1", sample["image"][:2]).unsqueeze(0).to(device)
        if img_s1.shape[-1] != 224 or img_s1.shape[-2] != 224:
            img_s1_224 = F.interpolate(img_s1, size=(224, 224), mode="bilinear", align_corners=False)
        else:
            img_s1_224 = img_s1

        with torch.no_grad():
            t0 = time.perf_counter_ns()
            feats = saber.backbone(img_s1_224, saber.s1_wvs)
            z1 = saber.s1_projection(feats)
            t1 = time.perf_counter_ns()
            lat_bb = (t1 - t0) / 1e6

            embs_per_step = {}
            lat_ode_per_step = {}

            for steps in ode_steps_list:
                t2 = time.perf_counter_ns()
                original_steps = saber.bridge.ode_steps
                saber.bridge.ode_steps = steps
                z_q = saber.bridge(z1)
                saber.bridge.ode_steps = original_steps
                emb = saber.retrieval_head(z_q).cpu().numpy()[0]
                t3 = time.perf_counter_ns()

                embs_per_step[steps] = emb
                lat_ode_per_step[steps] = (t3 - t2) / 1e6

        query_data.append({
            "label": q_label,
            "embs": embs_per_step,
            "lat_bb": lat_bb,
            "lat_ode": lat_ode_per_step
        })

    # Part 1: Ablation over Rerank Shortlist K (holding ode_steps = 5 fixed)
    print("\n--- PART 1: SHORTLIST K ABLATION (ode_steps = 5) ---")
    results_shortlist = []
    
    for sk in shortlist_k_list:
        retriever = Retriever(
            index=idx_saber,
            gallery_names=gallery_names,
            gallery_labels=gallery_labels,
            gallery_embeddings=gallery_embs_saber,
            rerank_enabled=True,
            rerank_shortlist_k=sk,
            rerank_neighbor_k=10
        )

        prec_list, rec_list, f1_list, jaccard_list, lat_vector_list, lat_rerank_list = [], [], [], [], [], []

        for q in query_data:
            q_label = q["label"]
            emb = q["embs"][5]

            t0 = time.perf_counter_ns()
            # We measure retrieve latency breakdown
            matches = retriever.retrieve(emb, k=5, query_label=q_label)
            t1 = time.perf_counter_ns()

            lat_total_retrieval = (t1 - t0) / 1e6

            # Compute metrics for top-5
            jaccards = [calculate_jaccard(q_label, m["label"]) for m in matches[:5]]
            top5_labels = [m["label"] for m in matches[:5]]

            p5 = sum(1 for j in jaccards if j > 0.0) / 5.0
            r5 = calculate_recall_class_coverage(q_label, top5_labels)
            f15 = (2 * p5 * r5 / (p5 + r5)) if (p5 + r5) > 0 else 0.0
            j5 = float(np.mean(jaccards))

            prec_list.append(p5)
            rec_list.append(r5)
            f1_list.append(f15)
            jaccard_list.append(j5)
            lat_rerank_list.append(lat_total_retrieval)

        mean_p5 = float(np.mean(prec_list)) * 100
        mean_r5 = float(np.mean(rec_list)) * 100
        mean_f15 = float(np.mean(f1_list)) * 100
        mean_j5 = float(np.mean(jaccard_list)) * 100
        mean_lat = float(np.mean(lat_rerank_list))

        results_shortlist.append({
            "shortlist_k": sk,
            "precision_5": mean_p5,
            "recall_5": mean_r5,
            "f1_5": mean_f15,
            "jaccard_5": mean_j5,
            "retrieval_lat_ms": mean_lat
        })
        print(f"Shortlist K={sk:3d} | Prec@5={mean_p5:.2f}% | Rec@5={mean_r5:.2f}% | F1@5={mean_f15:.2f}% | Jaccard@5={mean_j5:.2f}% | Retrieval Latency={mean_lat:.2f} ms")

    # Part 2: Ablation over ODE Steps (holding shortlist_k = 30 fixed)
    print("\n--- PART 2: NEURAL ODE STEPS ABLATION (shortlist_k = 30) ---")
    results_ode = []
    
    retriever_30 = Retriever(
        index=idx_saber,
        gallery_names=gallery_names,
        gallery_labels=gallery_labels,
        gallery_embeddings=gallery_embs_saber,
        rerank_enabled=True,
        rerank_shortlist_k=30,
        rerank_neighbor_k=10
    )

    for steps in ode_steps_list:
        prec_list, rec_list, f1_list, jaccard_list, lat_ode_list, lat_tot_list = [], [], [], [], [], []

        for q in query_data:
            q_label = q["label"]
            emb = q["embs"][steps]
            lat_bb = q["lat_bb"]
            lat_ode = q["lat_ode"][steps]

            t0 = time.perf_counter_ns()
            matches = retriever_30.retrieve(emb, k=5, query_label=q_label)
            t1 = time.perf_counter_ns()
            lat_ret = (t1 - t0) / 1e6

            jaccards = [calculate_jaccard(q_label, m["label"]) for m in matches[:5]]
            top5_labels = [m["label"] for m in matches[:5]]

            p5 = sum(1 for j in jaccards if j > 0.0) / 5.0
            r5 = calculate_recall_class_coverage(q_label, top5_labels)
            f15 = (2 * p5 * r5 / (p5 + r5)) if (p5 + r5) > 0 else 0.0
            j5 = float(np.mean(jaccards))

            prec_list.append(p5)
            rec_list.append(r5)
            f1_list.append(f15)
            jaccard_list.append(j5)
            lat_ode_list.append(lat_ode)
            lat_tot_list.append(lat_bb + lat_ode + lat_ret)

        mean_p5 = float(np.mean(prec_list)) * 100
        mean_r5 = float(np.mean(rec_list)) * 100
        mean_f15 = float(np.mean(f1_list)) * 100
        mean_j5 = float(np.mean(jaccard_list)) * 100
        mean_ode_lat = float(np.mean(lat_ode_list))
        mean_tot_lat = float(np.mean(lat_tot_list))

        results_ode.append({
            "ode_steps": steps,
            "precision_5": mean_p5,
            "recall_5": mean_r5,
            "f1_5": mean_f15,
            "jaccard_5": mean_j5,
            "ode_lat_ms": mean_ode_lat,
            "total_lat_ms": mean_tot_lat
        })
        print(f"ODE Steps={steps:2d} | Prec@5={mean_p5:.2f}% | Rec@5={mean_r5:.2f}% | F1@5={mean_f15:.2f}% | Jaccard@5={mean_j5:.2f}% | ODE Latency={mean_ode_lat:.2f} ms | Total Latency={mean_tot_lat:.2f} ms")

    # Compile Markdown Report
    report = []
    report.append("# SABER Comprehensive Empirical Ablation Study — Latency vs Accuracy Analysis\n")
    report.append("### Scientific Evaluation Report (ISRO BAH 2026 Grand Finale)\n")
    report.append(f"- **Evaluated Queries**: 500 Test Scenes from BEN-14K (Sentinel-1 SAR → Sentinel-2 Optical)")
    report.append(f"- **Hardware**: CUDA Acceleration (NVIDIA GeForce RTX 4050)")
    report.append(f"- **Default Architecture**: Re-ranking enabled across all experiments\n")

    report.append("## 1. Shortlist K Ablation (`shortlist_k` vs Latency & Accuracy)\n")
    report.append("Holding Neural ODE steps fixed at `ode_steps = 5`, we evaluate `shortlist_k ∈ {10, 15, 20, 25, 30, 40, 50, 75, 100}`:\n")
    report.append("| Shortlist K | Precision@5 (%) | Recall@5 (%) | F1-Score@5 (%) | Mean Jaccard@5 (%) | Retrieval Latency (ms) |")
    report.append("| :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results_shortlist:
        report.append(f"| **{r['shortlist_k']}** | {r['precision_5']:.2f}% | {r['recall_5']:.2f}% | **{r['f1_5']:.2f}%** | {r['jaccard_5']:.2f}% | **{r['retrieval_lat_ms']:.2f} ms** |")

    report.append("\n## 2. Neural ODE Steps Ablation (`ode_steps` vs Latency & Accuracy)\n")
    report.append("Holding `shortlist_k = 30` fixed, we evaluate Euler solver steps `ode_steps ∈ {1, 2, 3, 4, 5, 8, 10}`:\n")
    report.append("| ODE Steps | Precision@5 (%) | Recall@5 (%) | F1-Score@5 (%) | Mean Jaccard@5 (%) | ODE Solver Latency (ms) | Total Pipeline Latency (ms) |")
    report.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in results_ode:
        report.append(f"| **{r['ode_steps']}** | {r['precision_5']:.2f}% | {r['recall_5']:.2f}% | **{r['f1_5']:.2f}%** | {r['jaccard_5']:.2f}% | **{r['ode_lat_ms']:.2f} ms** | **{r['total_lat_ms']:.2f} ms** |")

    report.append("\n## 3. Empirical Findings & Pareto Frontier Recommendation\n")
    report.append("Based on the 500-query benchmark results:\n")

    best_shortlist = max(results_shortlist, key=lambda x: x["f1_5"])
    report.append(f"- **Optimal Shortlist K**: `shortlist_k = {best_shortlist['shortlist_k']}` delivers F1@5 = **{best_shortlist['f1_5']:.2f}%** and Jaccard@5 = **{best_shortlist['jaccard_5']:.2f}%** with retrieval latency of **{best_shortlist['retrieval_lat_ms']:.2f} ms**.")

    report_text = "\n".join(report)

    os.makedirs("docs", exist_ok=True)
    out_path = "docs/saber_ablation_study_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nEmpirical Ablation Report successfully saved to '{out_path}'!")

if __name__ == "__main__":
    main()
