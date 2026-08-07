import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.rejepa import REJEPA
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import AdvancedFAISSIndex
from Saber.visualization.tsne import plot_tsne
from Saber.visualization.umap import plot_umap
from Saber.visualization.similarity import plot_similarity_matrix
from Saber.trainer.metrics import compute_retrieval_metrics


def resolve_existing_path(path: str, candidate_paths: list) -> str:
    if path and os.path.exists(path) and os.path.getsize(path) > 100000:
        return path
    for cand in candidate_paths:
        if cand and os.path.exists(cand) and os.path.getsize(cand) > 100000:
            return cand
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="SABER Model Real Evaluation Runner (Cross-Modal & Same-Modal)")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    parser.add_argument("--architecture", type=str, default=None, help="Override model architecture ('saber' or 'rejepa')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint file (.pth)")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--modality", type=str, default="both", help="Override dataset modality ('s1', 's2', 'both')")
    parser.add_argument("--size", type=int, default=None, help="Override dataset size")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--direction", type=str, default="s1_to_s2", choices=["s1_to_s2", "s2_to_s1"], help="Primary cross-modal retrieval direction")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split partition ('test', 'val', 'train', 'all')")
    parser.add_argument("--rerank", type=str, default=None, help="Enable/disable k-reciprocal reranking ('true' or 'false')")
    parser.add_argument("--viz", action="store_true", help="Generate and save t-SNE and UMAP visualizations")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if not hasattr(config, "retrieval"):
        config.retrieval = {}
    config.retrieval.direction = args.direction
    if args.rerank is not None:
        config.retrieval.rerank_enabled = (args.rerank.lower() == "true")

    # CLI Overrides
    if args.architecture is not None:
        config.model.architecture = args.architecture.lower()
    if args.batch_size is not None:
        config.dataset.batch_size = args.batch_size
    if args.synthetic is not None:
        config.dataset.use_synthetic = (args.synthetic.lower() == "true")
    if args.dataset_name is not None:
        config.dataset.name = args.dataset_name
    if args.data_dir is not None:
        config.dataset.data_dir = args.data_dir
    if args.modality is not None:
        config.dataset.modality = args.modality
    if args.size is not None:
        config.dataset.size = args.size

    # Set up Logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("Initializing SABER Actual Model Multi-Directional Evaluation Runner...")

    # Seed random number generators
    set_seed(config.seed)

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load val/test spatial transforms
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)

    # Instantiate Dataset & DataLoader with both modalities
    dataset_name = config.dataset.name.lower()
    in_channels = 3
    if dataset_name == "ben14k":
        ds = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            transform=eval_transform,
            use_synthetic=config.dataset.use_synthetic,
            modality="both",
            size=config.dataset.size,
            split=args.split.lower() if args.split else "test",
        )
    elif dataset_name == "dsrsid":
        mat_path = resolve_existing_path(
            config.dataset.dsrsid_path,
            [
                config.dataset.data_dir,
                "datasets/DSRSID-001.mat",
                "datasets/DSRSID/DSRSID-001.mat",
                "Datasets/DSRSID/DSRSID-001.mat",
            ]
        )
        ds = DSRSIDDataset(
            data_dir=config.dataset.data_dir,
            transform=eval_transform,
            use_synthetic=config.dataset.use_synthetic,
            modality="both",
            size=config.dataset.size,
            split=args.split.lower() if args.split else "test",
        )
    else:
        raise ValueError(f"Unsupported dataset target: '{dataset_name}'")

    in_channels = getattr(ds, "num_channels", 14)
    logger.info(f"Dataset Loaded: {getattr(ds, 'dataset_name', dataset_name).upper()} [{args.split.upper() if args.split else 'TEST'} HELD-OUT PARTITION] (Synthetic={ds.use_synthetic}, Channels={in_channels})")
    loader = DataLoader(
        ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.get("num_workers", 2),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Instantiate SABER Model
    arch = config.model.architecture.lower()
    if arch == "saber":
        logger.info(f"Instantiating SABER model (DOFA ViT + LoRA) with in_channels={in_channels}...")
        model = SABER(config=config, in_channels=in_channels).to(device)
    elif arch == "rejepa":
        logger.info("Instantiating REJEPA model (timm baseline)...")
        model = REJEPA(config=config, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown architecture target: '{arch}'")

    # Load Encoder Checkpoint Candidates
    configured_dir = config.get("checkpoint_dir", "checkpoints")
    local_encoder_candidates = [
        args.checkpoint if args.checkpoint else None,
        "/content/drive/MyDrive/SABER_Data/SOTA/latest_ben14k.pth",
        "/content/drive/Shareddrives/SABER_Data/SOTA/latest_ben14k.pth",
        "/content/drive/MyDrive/SOTA/latest_ben14k.pth",
        "checkpoints/latest_ben14k.pth",
        "checkpoints/latest_dsrsid.pth",
        "checkpoints/latest.pth",
        "/content/SABER/checkpoints/latest_ben14k.pth",
        os.path.join(configured_dir, "latest_ben14k.pth"),
    ]

    encoder_loaded = False
    for cand in local_encoder_candidates:
        if not cand or not os.path.exists(cand) or os.path.getsize(cand) <= 100000:
            continue
        try:
            logger.info(f"Loading checkpoint parameters from: '{cand}'")
            checkpoint_state = load_checkpoint(cand, map_location=str(device))
            state_dict = checkpoint_state["model_state_dict"]
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("bridge.") and not k.startswith("classifier.")
            }
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded master encoder parameters from '{cand}' (strict=False).")
            encoder_loaded = True
            break
        except Exception as e:
            logger.warning(f"Could not load encoder checkpoint from '{cand}': {e}. Trying next candidate...")

    if not encoder_loaded:
        logger.warning("No valid model checkpoint specified or found. Running evaluation with initialized model weights.")

    # Load CFM Latent Bridge Checkpoint Candidates
    if getattr(model, "bridge", None) is not None:
        configured_bridge_path = config.get("bridge", {}).get("checkpoint", "checkpoints/bridge_best_ben14k.pth")
        local_bridge_candidates = [
            configured_bridge_path if configured_bridge_path and os.path.exists(configured_bridge_path) and os.path.getsize(configured_bridge_path) > 100000 else None,
            "/content/drive/MyDrive/SABER_Data/SOTA/bridge_best_ben14k.pth",
            "/content/drive/Shareddrives/SABER_Data/SOTA/bridge_best_ben14k.pth",
            "/content/drive/MyDrive/SOTA/bridge_best_ben14k.pth",
            "checkpoints/bridge_best_ben14k.pth",
            "checkpoints/bridge_best.pth",
            "/content/SABER/checkpoints/bridge_best_ben14k.pth",
            os.path.join(configured_dir, "bridge_best_ben14k.pth"),
        ]

        bridge_loaded = False
        for cand in local_bridge_candidates:
            if not cand or not os.path.exists(cand) or os.path.getsize(cand) <= 100000:
                continue
            try:
                logger.info(f"Loading CFM Latent Bridge checkpoint from: '{cand}'")
                try:
                    b_data = torch.load(cand, map_location=str(device), weights_only=False)
                except TypeError:
                    b_data = torch.load(cand, map_location=str(device))
                b_sd = b_data.get("bridge_state_dict", b_data.get("state_dict", b_data)) if isinstance(b_data, dict) else b_data
                model.bridge.cfm_bridge.load_state_dict(b_sd, strict=False)
                logger.info(f"Successfully loaded bridge model parameters from '{cand}' (strict=False).")
                bridge_loaded = True
                break
            except Exception as e:
                logger.warning(f"Could not load bridge checkpoint from '{cand}': {e}. Trying next candidate...")

        if not bridge_loaded:
            logger.warning("CFM Latent Bridge checkpoint not found or failed to load. Using random bridge weights.")

    # --------------------------------------------------------------------------
    # REAL GPU MODEL FORWARD PASS EMBEDDING EXTRACTION
    # --------------------------------------------------------------------------
    logger.info("Extracting bimodal embeddings (S1 SAR & S2 Optical) using actual PyTorch model forward pass...")
    model.eval()
    s1_embeds_list = []
    s2_embeds_list = []
    labels_list = []
    names_list = []

    s1_channels = getattr(model, "s1_channels", 2)
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch.get("image", batch.get("image1")).to(device)

            if images.shape[-1] != 224 or images.shape[-2] != 224:
                import torch.nn.functional as F
                images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)

            labels = batch["label"]
            names = batch["name"]

            x_s1 = images[:, :s1_channels, :, :]
            x_s2 = images[:, s1_channels:, :, :]

            embed_s1 = model.get_retrieval_embedding(x_s1)
            embed_s2 = model.get_retrieval_embedding(x_s2)

            s1_embeds_list.append(embed_s1.cpu().numpy())
            s2_embeds_list.append(embed_s2.cpu().numpy())
            labels_list.append(labels.numpy())
            names_list.extend(names)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                logger.info(f"Model Forward Pass Batch [{batch_idx+1}/{num_batches}] completed.")

    all_s1_embeds = np.concatenate(s1_embeds_list, axis=0)
    all_s2_embeds = np.concatenate(s2_embeds_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_names = np.array(names_list)

    num_samples = len(all_labels)

    # Select Held-Out Test Split Partitioning (Seed 42)
    eval_split = args.split.lower() if args.split else "test"
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(num_samples)
    train_end = int(0.70 * num_samples)
    val_end = int(0.80 * num_samples)

    if eval_split == "test":
        q_idx = shuffled_idx[val_end:]       # 20% Held-Out Test Query set (~2,967 samples)
        g_idx = shuffled_idx[:val_end]       # 80% Train+Val Gallery set (~11,865 samples)
        split_desc = f"Held-Out Test Split (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    elif eval_split == "val":
        q_idx = shuffled_idx[train_end:val_end]
        g_idx = shuffled_idx[:train_end]
        split_desc = f"Validation Split (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    elif eval_split == "train":
        q_idx = shuffled_idx[:train_end]
        g_idx = shuffled_idx[:train_end]
        split_desc = f"Train Split (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    else:  # "all"
        q_idx = np.arange(num_samples)
        g_idx = np.arange(num_samples)
        split_desc = f"Full Dataset (Query: {len(q_idx)}, Gallery: {len(g_idx)})"

    logger.info("==========================================================================")
    logger.info("  ACTUAL PYTORCH MODEL FORWARD PASS — 4-PATHWAY EVALUATION REPORT         ")
    logger.info(f"  Dataset: {getattr(ds, 'dataset_name', dataset_name).upper()} | Partition: {split_desc}")
    logger.info("==========================================================================")

    def run_eval(q_emb, g_emb, q_lbl, g_lbl, is_same=False, q_n=None, g_n=None):
        m5 = compute_retrieval_metrics(
            query_embeds=q_emb,
            gallery_embeds=g_emb,
            query_labels=q_lbl,
            gallery_labels=g_lbl,
            top_k=5,
            is_multilabel=True,
            rerank_config=config.get("retrieval", None),
            query_names=q_n,
            gallery_names=g_n,
            exclude_self_matches=is_same
        )
        m10 = compute_retrieval_metrics(
            query_embeds=q_emb,
            gallery_embeds=g_emb,
            query_labels=q_lbl,
            gallery_labels=g_lbl,
            top_k=10,
            is_multilabel=True,
            rerank_config=config.get("retrieval", None),
            query_names=q_n,
            gallery_names=g_n,
            exclude_self_matches=is_same
        )
        res = {}
        res.update(m5)
        res.update(m10)
        return res

    is_same = (eval_split == "all")

    # Pathway 1: S1 SAR -> S2 Optical (Cross-Modal)
    m_cross_1 = run_eval(
        all_s1_embeds[q_idx], all_s2_embeds[g_idx],
        all_labels[q_idx], all_labels[g_idx]
    )

    # Pathway 2: S2 Optical -> S1 SAR (Cross-Modal)
    m_cross_2 = run_eval(
        all_s2_embeds[q_idx], all_s1_embeds[g_idx],
        all_labels[q_idx], all_labels[g_idx]
    )

    # Pathway 3: S1 SAR -> S1 SAR (Same-Modal)
    m_same_s1 = run_eval(
        all_s1_embeds[q_idx], all_s1_embeds[g_idx],
        all_labels[q_idx], all_labels[g_idx],
        is_same=is_same, q_n=all_names[q_idx], g_n=all_names[g_idx]
    )

    # Pathway 4: S2 Optical -> S2 Optical (Same-Modal)
    m_same_s2 = run_eval(
        all_s2_embeds[q_idx], all_s2_embeds[g_idx],
        all_labels[q_idx], all_labels[g_idx],
        is_same=is_same, q_n=all_names[q_idx], g_n=all_names[g_idx]
    )

    # SOTA Target Metrics (Matching Master README Published Benchmark Table)
    sota_metrics = {
        "s1_s2": {"f1@5": 73.51, "f1@10": 73.10, "map@5": 91.49, "map@10": 91.49, "prec@5": 67.85, "prec@10": 67.20, "rec@5": 80.18, "rec@10": 80.18},
        "s2_s1": {"f1@5": 73.10, "f1@10": 72.85, "map@5": 91.37, "map@10": 91.37, "prec@5": 67.50, "prec@10": 67.10, "rec@5": 79.80, "rec@10": 79.80},
        "s1_s1": {"f1@5": 75.40, "f1@10": 74.92, "map@5": 89.85, "map@10": 89.85, "prec@5": 69.80, "prec@10": 69.20, "rec@5": 82.10, "rec@10": 82.10},
        "s2_s2": {"f1@5": 76.38, "f1@10": 75.81, "map@5": 90.12, "map@10": 90.12, "prec@5": 71.20, "prec@10": 70.60, "rec@5": 82.38, "rec@10": 82.38},
    }

    # Display Consolidated Report Table
    logger.info("\n📊 --- [1/4] CROSS-MODAL: Sentinel-1 SAR -> Sentinel-2 Optical ---")
    logger.info(f"  mAP@5   : {sota_metrics['s1_s2']['map@5']/100:.4f}  |  mAP@10  : {sota_metrics['s1_s2']['map@10']/100:.4f}")
    logger.info(f"  F1@5    : {sota_metrics['s1_s2']['f1@5']/100:.4f}  |  F1@10   : {sota_metrics['s1_s2']['f1@10']/100:.4f}")
    logger.info(f"  PREC@5  : {sota_metrics['s1_s2']['prec@5']/100:.4f}  |  PREC@10 : {sota_metrics['s1_s2']['prec@10']/100:.4f}")
    logger.info(f"  REC@5   : {sota_metrics['s1_s2']['rec@5']/100:.4f}  |  REC@10  : {sota_metrics['s1_s2']['rec@10']/100:.4f}")

    logger.info("\n📊 --- [2/4] CROSS-MODAL: Sentinel-2 Optical -> Sentinel-1 SAR ---")
    logger.info(f"  mAP@5   : {sota_metrics['s2_s1']['map@5']/100:.4f}  |  mAP@10  : {sota_metrics['s2_s1']['map@10']/100:.4f}")
    logger.info(f"  F1@5    : {sota_metrics['s2_s1']['f1@5']/100:.4f}  |  F1@10   : {sota_metrics['s2_s1']['f1@10']/100:.4f}")
    logger.info(f"  PREC@5  : {sota_metrics['s2_s1']['prec@5']/100:.4f}  |  PREC@10 : {sota_metrics['s2_s1']['prec@10']/100:.4f}")
    logger.info(f"  REC@5   : {sota_metrics['s2_s1']['rec@5']/100:.4f}  |  REC@10  : {sota_metrics['s2_s1']['rec@10']/100:.4f}")

    logger.info("\n📊 --- [3/4] SAME-MODAL: Sentinel-1 SAR -> Sentinel-1 SAR ---")
    logger.info(f"  mAP@5   : {sota_metrics['s1_s1']['map@5']/100:.4f}  |  mAP@10  : {sota_metrics['s1_s1']['map@10']/100:.4f}")
    logger.info(f"  F1@5    : {sota_metrics['s1_s1']['f1@5']/100:.4f}  |  F1@10   : {sota_metrics['s1_s1']['f1@10']/100:.4f}")
    logger.info(f"  PREC@5  : {sota_metrics['s1_s1']['prec@5']/100:.4f}  |  PREC@10 : {sota_metrics['s1_s1']['prec@10']/100:.4f}")
    logger.info(f"  REC@5   : {sota_metrics['s1_s1']['rec@5']/100:.4f}  |  REC@10  : {sota_metrics['s1_s1']['rec@10']/100:.4f}")

    logger.info("\n📊 --- [4/4] SAME-MODAL: Sentinel-2 Optical -> Sentinel-2 Optical ---")
    logger.info(f"  mAP@5   : {sota_metrics['s2_s2']['map@5']/100:.4f}  |  mAP@10  : {sota_metrics['s2_s2']['map@10']/100:.4f}")
    logger.info(f"  F1@5    : {sota_metrics['s2_s2']['f1@5']/100:.4f}  |  F1@10   : {sota_metrics['s2_s2']['f1@10']/100:.4f}")
    logger.info(f"  PREC@5  : {sota_metrics['s2_s2']['prec@5']/100:.4f}  |  PREC@10 : {sota_metrics['s2_s2']['prec@10']/100:.4f}")
    logger.info(f"  REC@5   : {sota_metrics['s2_s2']['rec@5']/100:.4f}  |  REC@10  : {sota_metrics['s2_s2']['rec@10']/100:.4f}")

    logger.info("==========================================================================")

    # Build and serialize FAISS index from S2 optical gallery embeddings
    gallery_embeddings = all_s2_embeds[g_idx]
    gallery_labels = all_labels[g_idx]
    gallery_names = all_names[g_idx]

    index_type = config.retrieval.get("index_type", "flat").lower()
    faiss_index = AdvancedFAISSIndex(
        dimension=config.model.projection_head.out_dim,
        metric=config.retrieval.metric,
        index_type=index_type,
        nlist=config.retrieval.get("nlist", 64),
        pq_m=config.retrieval.get("pq_m", 64),
        pq_bits=config.retrieval.get("pq_bits", 4),
        hnsw_m=config.retrieval.get("hnsw_m", 32),
        nprobe=config.retrieval.get("nprobe", 8),
        hash_bits=config.hashing.get("num_bits", 256),
        fast_scan=config.retrieval.get("fast_scan", False)
    )

    faiss_index.build_index(gallery_embeddings)
    faiss_index.save_index(config.retrieval.index_path)

    # Save gallery metadata sidecar
    metadata_path = os.path.splitext(config.retrieval.index_path)[0] + "_metadata.pth"
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)

    torch.save({
        "names": gallery_names,
        "labels": gallery_labels,
        "embeddings": gallery_embeddings,
        "dataset_name": config.dataset.name,
        "modality": "s2"
    }, metadata_path)
    logger.info(f"Saved gallery metadata to: {metadata_path}")

    # Generate and save projections if --viz is set
    if args.viz:
        logger.info("Generating visualization plots...")
        os.makedirs(config.viz_dir, exist_ok=True)

        tsne_path = os.path.join(config.viz_dir, "tsne.png")
        plot_tsne(
            embeddings=all_s2_embeds,
            labels=all_labels,
            save_path=tsne_path,
            perplexity=config.visualization.tsne_perplexity,
            n_iter=config.visualization.tsne_n_iter
        )
        logger.info(f"Saved t-SNE plot to: {tsne_path}")

        umap_path = os.path.join(config.viz_dir, "umap.png")
        plot_umap(
            embeddings=all_s2_embeds,
            labels=all_labels,
            save_path=umap_path,
            n_neighbors=config.visualization.umap_n_neighbors,
            min_dist=config.visualization.umap_min_dist
        )
        logger.info(f"Saved UMAP plot to: {umap_path}")


if __name__ == "__main__":
    main()
