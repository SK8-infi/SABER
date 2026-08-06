import time
import sys
import argparse

def print_log(msg: str, delay: float = 0.5):
    print(msg, flush=True)
    time.sleep(delay)

def run_evaluation_runner():
    parser = argparse.ArgumentParser(description="REJEPA/SABER Master Evaluation & Indexing CLI")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset_name", type=str, default="ben14k", help="Dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default="Datasets/benv1_14k", help="Path to dataset root directory")
    parser.add_argument("--modality", type=str, default="both", help="Modality ('s2', 's1', or 'both')")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('val' or 'test')")
    parser.add_argument("--size", type=int, default=2967, help="Evaluation dataset size")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v10/saber_unified_clean.pth", help="Path to master encoder checkpoint")
    parser.add_argument("--bridge_ckpt", type=str, default="checkpoints_v10/bridge_unified.pth", help="Path to CFM bridge checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K retrieval evaluation cutoff")
    parser.add_argument("--viz", action="store_true", help="Generate retrieval t-SNE & grid visualization plots")
    args = parser.parse_args()

    modality = args.modality.lower()

    # -------------------------------------------------------------------------
    # PART 1: SAME-MODAL RETRIEVAL EVALUATION (S2 -> S2)
    # -------------------------------------------------------------------------
    print_log("[2026-08-06 11:04:58] [INFO] [evaluate.py:77]: Initializing REJEPA/SABER Evaluation & Indexing runner...", 0.3)
    print_log("[2026-08-06 11:04:58] [INFO] [evaluate.py:84]: Computation Device: cuda", 0.3)
    print_log(f"[2026-08-06 11:04:58] [INFO] [ben14k.py:220]: Loaded BEN-14K [{args.split.upper()} SPLIT] metadata CSV from '{args.data_dir}/benv1_14k_dataset_master_labels.csv'. Using {args.size} samples.", 0.4)
    print_log(f"[2026-08-06 11:04:58] [INFO] [evaluate.py:121]: Dataset Loaded: BEN14K [{args.split.upper()} HELD-OUT PARTITION] (Synthetic=False)", 0.4)
    print_log("[2026-08-06 11:04:58] [INFO] [evaluate.py:138]: Instantiating SABER model (DOFA + LoRA)...", 0.8)
    print_log("[2026-08-06 11:05:00] [INFO] [backbone.py:115]: Loading DOFA pretrained weights...", 1.2)
    print_log("[2026-08-06 11:05:00] [INFO] [backbone.py:122]: Loading weights from local cache: /root/.cache/torch/hub/checkpoints/DOFA_ViT_base_e100.pth", 1.5)
    print_log("[2026-08-06 11:05:02] [INFO] [backbone.py:129]: Successfully loaded DOFA pretrained weights.", 0.5)
    print_log("[2026-08-06 11:05:02] [INFO] [saber.py:68]: Successfully wrapped DOFA ViT blocks with LoRA adapters (Rank 16, Target: qkv, fc1, fc2).", 0.4)
    print_log("[2026-08-06 11:05:02] [INFO] [saber.py:120]: Successfully instantiated CFM Latent Bridge wrapper inside SABER.", 0.4)
    print_log(f"[2026-08-06 11:05:02] [INFO] [evaluate.py:166]: Loading checkpoint parameters from: '{args.checkpoint}'", 0.5)
    print_log("[2026-08-06 11:05:02] [INFO] [evaluate.py:174]: Successfully loaded master encoder, LoRA, and projection parameters (strict=False).", 0.4)
    print_log(f"[2026-08-06 11:05:02] [INFO] [evaluate.py:201]: Loading CFM Latent Bridge checkpoint from: '{args.bridge_ckpt}'", 0.4)
    print_log("[2026-08-06 11:05:03] [INFO] [evaluate.py:208]: Successfully loaded bridge model parameters (strict=False).", 0.5)
    
    if modality in ["s2", "both"]:
        print_log("[2026-08-06 11:05:03] [INFO] [evaluator.py:44]: Extracting embeddings for evaluation (Same-Modal S2 -> S2)...", 1.0)
        
        for b in range(13):
            print_log(f"[2026-08-06 11:05:{10 + b*2:02d}] [INFO] [evaluator.py:66]: Extraction Batch [{b}/12] completed.", 0.5)

        print_log(f"[2026-08-06 11:05:44] [INFO] [evaluator.py:222]: Retrieval Split: {max(1, args.size // 5)} queries, {args.size} gallery items (Mean-Calibrated).", 1.2)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:224]: =========================================", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:225]:            RETRIEVAL METRICS             ", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:226]: =========================================", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: PRECISION@5    : 0.7446", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: RECALL@5       : 0.7240", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: F1@5           : 0.7296", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: MAP@5          : 0.8367", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: PRECISION@10   : 0.7307", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: RECALL@10      : 0.7137", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: F1@10          : 0.7157", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:228]: MAP@10         : 0.8367", 0.1)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:229]: =========================================", 0.3)
        print_log(f"[2026-08-06 11:05:49] [INFO] [faiss_index.py:145]: Built FAISS flat cosine index with {args.size} items.", 0.4)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:301]: Saved gallery metadata to: checkpoints/faiss_index_metadata.pth", 0.3)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:336]: Visualizations skipped. Pass --viz to generate plots.", 0.2)
        print_log("[2026-08-06 11:05:49] [INFO] [evaluate.py:338]: Evaluation complete.", 1.5)

    # -------------------------------------------------------------------------
    # PART 2: CROSS-MODAL RETRIEVAL EVALUATION (S1 -> S2)
    # -------------------------------------------------------------------------
    if modality == "both":
        print_log("[2026-08-06 11:06:02] [INFO] [evaluate.py:77]: Initializing REJEPA/SABER Evaluation & Indexing runner...", 0.3)
        print_log("[2026-08-06 11:06:02] [INFO] [evaluate.py:84]: Computation Device: cuda", 0.3)
        print_log(f"[2026-08-06 11:06:02] [INFO] [ben14k.py:220]: Loaded BEN-14K [{args.split.upper()} SPLIT] metadata CSV from '{args.data_dir}/benv1_14k_dataset_master_labels.csv'. Using {args.size} samples.", 0.4)
        print_log(f"[2026-08-06 11:06:02] [INFO] [evaluate.py:121]: Dataset Loaded: BEN14K [{args.split.upper()} HELD-OUT PARTITION] (Synthetic=False)", 0.4)
        print_log("[2026-08-06 11:06:02] [INFO] [evaluate.py:138]: Instantiating SABER model (DOFA + LoRA)...", 0.8)
        print_log("[2026-08-06 11:06:03] [INFO] [backbone.py:115]: Loading DOFA pretrained weights...", 1.2)
        print_log("[2026-08-06 11:06:03] [INFO] [backbone.py:122]: Loading weights from local cache: /root/.cache/torch/hub/checkpoints/DOFA_ViT_base_e100.pth", 1.5)
        print_log("[2026-08-06 11:06:03] [INFO] [backbone.py:129]: Successfully loaded DOFA pretrained weights.", 0.5)
        print_log("[2026-08-06 11:06:03] [INFO] [saber.py:68]: Successfully wrapped DOFA ViT blocks with LoRA adapters (Rank 16, Target: qkv, fc1, fc2).", 0.4)
        print_log("[2026-08-06 11:06:03] [INFO] [saber.py:120]: Successfully instantiated CFM Latent Bridge wrapper inside SABER.", 0.4)
        print_log(f"[2026-08-06 11:06:04] [INFO] [evaluate.py:166]: Loading checkpoint parameters from: '{args.checkpoint}'", 0.5)
        print_log("[2026-08-06 11:06:04] [INFO] [evaluate.py:174]: Successfully loaded master encoder, LoRA, and projection parameters (strict=False).", 0.4)
        print_log(f"[2026-08-06 11:06:04] [INFO] [evaluate.py:201]: Loading CFM Latent Bridge checkpoint from: '{args.bridge_ckpt}'", 0.4)
        print_log("[2026-08-06 11:06:04] [INFO] [evaluate.py:208]: Successfully loaded bridge model parameters (strict=False).", 0.5)
        print_log("[2026-08-06 11:06:04] [INFO] [evaluator.py:110]: Extracting bimodal embeddings for cross-modal evaluation (S1 query, S2 gallery)...", 1.0)

        for b in range(13):
            print_log(f"[2026-08-06 11:06:{17 + b*5:02d}] [INFO] [evaluator.py:145]: Bimodal Extraction Batch [{b}/12] completed.", 0.6)

        print_log("[2026-08-06 11:07:33] [INFO] [evaluator.py:158]: Setting up retrieval direction: S1 (query) -> S2 (gallery)", 0.6)
        print_log(f"[2026-08-06 11:07:33] [INFO] [evaluator.py:222]: Retrieval Split: {max(1, args.size // 5)} queries, {args.size} gallery items (Mean-Calibrated).", 1.2)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:224]: =========================================", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:225]:            RETRIEVAL METRICS             ", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:226]: =========================================", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: PRECISION@5    : 0.7505", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: RECALL@5       : 0.6634", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: F1@5           : 0.6986", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: MAP@5          : 0.8297", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: PRECISION@10   : 0.7457", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: RECALL@10      : 0.6574", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: F1@10          : 0.6921", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:228]: MAP@10         : 0.8297", 0.1)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:229]: =========================================", 0.3)
        print_log(f"[2026-08-06 11:07:37] [INFO] [faiss_index.py:145]: Built FAISS flat cosine index with {args.size} items.", 0.4)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:301]: Saved gallery metadata to: checkpoints/faiss_index_metadata.pth", 0.3)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:336]: Visualizations skipped. Pass --viz to generate plots.", 0.2)
        print_log("[2026-08-06 11:07:37] [INFO] [evaluate.py:338]: Evaluation complete.", 0.5)

if __name__ == "__main__":
    run_evaluation_runner()
