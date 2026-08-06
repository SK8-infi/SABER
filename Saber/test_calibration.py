import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.ben14k import BEN14KDataset, BIGEARTHNET_19_CLASSES
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.trainer.metrics import compute_retrieval_metrics

def run_calibration_experiment(checkpoint_path: str = "checkpoints/latest_ben14k.pth", config_path: str = "Saber/configs/config.yaml"):
    print("="*80)
    print(" 🚀 MEAN-CENTERING & IDF-WEIGHTED CALIBRATION EXPERIMENT")
    print("="*80)
    
    config = load_config(config_path)
    set_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    print(f"Computation Device: {device}")
    
    # Load dataset
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)
    dataset = BEN14KDataset(
        data_dir=config.dataset.data_dir,
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
    
    # Load model
    model = SABER(config=config, in_channels=dataset.num_channels).to(device)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
        state_dict = ckpt["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.")}
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model weights.")
    
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
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"Extraction progress: Batch [{batch_idx + 1}/{num_batches}] ({((batch_idx + 1)/num_batches)*100:.1f}%)", flush=True)
                
    embeddings = np.concatenate(embeds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    names = np.array(names_list)
    num_samples = len(embeddings)
    
    # Split query vs gallery (20% / 80%)
    rng = np.random.RandomState(42)
    shuffled_indices = rng.permutation(num_samples)
    query_size = max(1, num_samples // 5)
    query_idx = np.sort(shuffled_indices[:query_size])
    gallery_idx = np.arange(num_samples)
    
    q_embeds_raw = embeddings[query_idx]
    q_labels = labels[query_idx]
    q_names = names[query_idx]
    
    g_embeds_raw = embeddings[gallery_idx]
    g_labels = labels[gallery_idx]
    g_names = names[gallery_idx]
    
    # --- METHOD 1: BASELINE (RAW EMBEDDINGS) ---
    print("\n" + "="*60)
    print(" 1. EVALUATING BASELINE (RAW UN-CALIBRATED EMBEDDINGS)")
    print("="*60)
    
    metrics_raw5 = compute_retrieval_metrics(
        query_embeds=q_embeds_raw,
        gallery_embeds=g_embeds_raw,
        query_labels=q_labels,
        gallery_labels=g_labels,
        top_k=5,
        is_multilabel=True,
        query_names=q_names,
        gallery_names=g_names,
        exclude_self_matches=True
    )
    metrics_raw10 = compute_retrieval_metrics(
        query_embeds=q_embeds_raw,
        gallery_embeds=g_embeds_raw,
        query_labels=q_labels,
        gallery_labels=g_labels,
        top_k=10,
        is_multilabel=True,
        query_names=q_names,
        gallery_names=g_names,
        exclude_self_matches=True
    )
    
    # Compute baseline dynamic range
    q_t = torch.tensor(q_embeds_raw, device=device)
    g_t = torch.tensor(g_embeds_raw, device=device)
    sims_raw = torch.matmul(q_t, g_t.t())
    mask_cpu = q_names[:, None] != g_names[None, :]
    mask = torch.tensor(mask_cpu, device=device)
    
    mean_sim_raw_all = torch.mean(sims_raw[mask]).item()
    
    q_lbl_t = torch.tensor(q_labels, dtype=torch.float32, device=device)
    g_lbl_t = torch.tensor(g_labels, dtype=torch.float32, device=device)
    intersection = torch.matmul(q_lbl_t, g_lbl_t.t())
    union = q_lbl_t.sum(dim=1, keepdim=True) + g_lbl_t.sum(dim=1, keepdim=True).t() - intersection
    jaccard = intersection / (union + 1e-8)
    
    pos_mask = (jaccard > 0.5) & mask
    mean_sim_raw_pos = torch.mean(sims_raw[pos_mask]).item()
    raw_gap = mean_sim_raw_pos - mean_sim_raw_all
    
    print(f"RAW F1@5                 : {metrics_raw5['f1@5']*100:.2f}%")
    print(f"RAW Precision@5          : {metrics_raw5['precision@5']*100:.2f}%")
    print(f"RAW Recall@5             : {metrics_raw5['recall@5']*100:.2f}%")
    print(f"RAW F1@10                : {metrics_raw10['f1@10']*100:.2f}%")
    print(f"RAW Random Pair Sim      : {mean_sim_raw_all:.4f}")
    print(f"RAW Match Pair Sim       : {mean_sim_raw_pos:.4f}")
    print(f"RAW Dynamic Range Gap    : {raw_gap:.4f}")
    
    # --- METHOD 2: MEAN-CENTERING CALIBRATION ---
    print("\n" + "="*60)
    print(" 2. EVALUATING MEAN-CENTERED CALIBRATION")
    print("="*60)
    
    # Compute global mean embedding vector over gallery
    mean_vec = np.mean(embeddings, axis=0, keepdims=True)
    
    embeddings_cal = embeddings - mean_vec
    norms = np.linalg.norm(embeddings_cal, axis=1, keepdims=True)
    embeddings_cal = embeddings_cal / np.maximum(norms, 1e-12)
    
    q_embeds_cal = embeddings_cal[query_idx]
    g_embeds_cal = embeddings_cal[gallery_idx]
    
    metrics_cal5 = compute_retrieval_metrics(
        query_embeds=q_embeds_cal,
        gallery_embeds=g_embeds_cal,
        query_labels=q_labels,
        gallery_labels=g_labels,
        top_k=5,
        is_multilabel=True,
        query_names=q_names,
        gallery_names=g_names,
        exclude_self_matches=True
    )
    metrics_cal10 = compute_retrieval_metrics(
        query_embeds=q_embeds_cal,
        gallery_embeds=g_embeds_cal,
        query_labels=q_labels,
        gallery_labels=g_labels,
        top_k=10,
        is_multilabel=True,
        query_names=q_names,
        gallery_names=g_names,
        exclude_self_matches=True
    )
    
    # Compute calibrated dynamic range
    q_t_c = torch.tensor(q_embeds_cal, device=device)
    g_t_c = torch.tensor(g_embeds_cal, device=device)
    sims_cal = torch.matmul(q_t_c, g_t_c.t())
    
    mean_sim_cal_all = torch.mean(sims_cal[mask]).item()
    mean_sim_cal_pos = torch.mean(sims_cal[pos_mask]).item()
    cal_gap = mean_sim_cal_pos - mean_sim_cal_all
    
    print(f"CALIBRATED F1@5          : {metrics_cal5['f1@5']*100:.2f}%  (Change: {(metrics_cal5['f1@5'] - metrics_raw5['f1@5'])*100:+.2f}%)")
    print(f"CALIBRATED Precision@5   : {metrics_cal5['precision@5']*100:.2f}%  (Change: {(metrics_cal5['precision@5'] - metrics_raw5['precision@5'])*100:+.2f}%)")
    print(f"CALIBRATED Recall@5      : {metrics_cal5['recall@5']*100:.2f}%  (Change: {(metrics_cal5['recall@5'] - metrics_raw5['recall@5'])*100:+.2f}%)")
    print(f"CALIBRATED F1@10         : {metrics_cal10['f1@10']*100:.2f}% (Change: {(metrics_cal10['f1@10'] - metrics_raw10['f1@10'])*100:+.2f}%)")
    print(f"CALIBRATED Random Pair   : {mean_sim_cal_all:.4f}")
    print(f"CALIBRATED Match Pair    : {mean_sim_cal_pos:.4f}")
    print(f"CALIBRATED Dynamic Range : {cal_gap:.4f}  (Expanded by {cal_gap/max(raw_gap, 1e-6):.1f}x!)")

    # --- SUMMARY MATRIX ---
    print("\n" + "="*80)
    print(" 📊 SUMMARY COMPARISON TABLE")
    print("="*80)
    print(f"{'Metric':<25} | {'Raw Baseline':<15} | {'Mean-Centered':<15} | {'Absolute Lift':<15}")
    print("-" * 75)
    print(f"{'F1@5':<25} | {metrics_raw5['f1@5']*100:<14.2f}% | {metrics_cal5['f1@5']*100:<14.2f}% | {(metrics_cal5['f1@5'] - metrics_raw5['f1@5'])*100:<+14.2f}%")
    print(f"{'Precision@5':<25} | {metrics_raw5['precision@5']*100:<14.2f}% | {metrics_cal5['precision@5']*100:<14.2f}% | {(metrics_cal5['precision@5'] - metrics_raw5['precision@5'])*100:<+14.2f}%")
    print(f"{'Recall@5':<25} | {metrics_raw5['recall@5']*100:<14.2f}% | {metrics_cal5['recall@5']*100:<14.2f}% | {(metrics_cal5['recall@5'] - metrics_raw5['recall@5'])*100:<+14.2f}%")
    print(f"{'F1@10':<25} | {metrics_raw10['f1@10']*100:<14.2f}% | {metrics_cal10['f1@10']*100:<14.2f}% | {(metrics_cal10['f1@10'] - metrics_raw10['f1@10'])*100:<+14.2f}%")
    print(f"{'Dynamic Range Gap':<25} | {raw_gap:<15.4f} | {cal_gap:<15.4f} | {(cal_gap - raw_gap):<+15.4f}")
    print("="*80)

if __name__ == "__main__":
    run_calibration_experiment()
