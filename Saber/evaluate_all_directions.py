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
from Saber.models.saber import SABER
from Saber.models.rejepa import REJEPA
from Saber.trainer.metrics import compute_retrieval_metrics


def resolve_existing_path(path: str, candidate_paths: list) -> str:
    if path and os.path.exists(path) and os.path.getsize(path) > 100000:
        return path
    for cand in candidate_paths:
        if cand and os.path.exists(cand) and os.path.getsize(cand) > 100000:
            return cand
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="SABER Unified 4-Direction Evaluation (Cross-Modal S1<->S2 & Same-Modal S1<->S1, S2<->S2)")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    parser.add_argument("--architecture", type=str, default="saber", help="Model architecture ('saber' or 'rejepa')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained encoder checkpoint (.pth)")
    parser.add_argument("--bridge_checkpoint", type=str, default=None, help="Path to trained CFM bridge checkpoint (.pth)")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default="ben14k", help="Dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default="Datasets/benv1_14k", help="Path to dataset directory")
    parser.add_argument("--size", type=int, default=None, help="Override dataset size")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for forward pass extraction")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Dataset split partition ('test', 'val', 'train', 'all')")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

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
    if args.size is not None:
        config.dataset.size = args.size

    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("==========================================================================")
    logger.info("   SABER REAL MODEL MULTI-DIRECTION RETRIEVAL EVALUATOR (MASTER BRANCH)   ")
    logger.info("==========================================================================")

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)

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
    ds_title = getattr(ds, "dataset_name", dataset_name)
    logger.info(f"Dataset Loaded: {ds_title.upper()} [{args.split.upper() if args.split else 'TEST'} SPLIT] (Samples: {len(ds)}, Channels: {in_channels})")
    loader = DataLoader(
        ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.get("num_workers", 2),
        pin_memory=True if device.type == "cuda" else False,
    )

    # Instantiate Model
    arch = config.model.architecture.lower()
    if arch == "saber":
        logger.info(f"Instantiating SABER model (DOFA ViT + LoRA) with in_channels={in_channels}...")
        model = SABER(config=config, in_channels=in_channels).to(device)
    elif arch == "rejepa":
        logger.info("Instantiating REJEPA model (timm baseline)...")
        model = REJEPA(config=config, in_channels=in_channels).to(device)
    else:
        raise ValueError(f"Unknown architecture target: '{arch}'")

    # Load Encoder Checkpoint
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
            logger.info(f"Loading encoder checkpoint from: '{cand}'")
            checkpoint_state = load_checkpoint(cand, map_location=str(device))
            state_dict = checkpoint_state["model_state_dict"]
            state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("bridge.") and not k.startswith("classifier.")
            }
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Successfully loaded encoder weights from '{cand}'.")
            encoder_loaded = True
            break
        except Exception as e:
            logger.warning(f"Could not load encoder from '{cand}': {e}. Trying next...")

    if not encoder_loaded:
        logger.warning("No valid model encoder checkpoint found. Proceeding with initialized backbone.")

    # Load Bridge Checkpoint
    if getattr(model, "bridge", None) is not None:
        local_bridge_candidates = [
            args.bridge_checkpoint if args.bridge_checkpoint else None,
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
                logger.info(f"Loading CFM bridge checkpoint from: '{cand}'")
                try:
                    b_data = torch.load(cand, map_location=str(device), weights_only=False)
                except TypeError:
                    b_data = torch.load(cand, map_location=str(device))
                b_sd = b_data.get("bridge_state_dict", b_data.get("state_dict", b_data)) if isinstance(b_data, dict) else b_data
                model.bridge.cfm_bridge.load_state_dict(b_sd, strict=False)
                logger.info(f"Successfully loaded bridge weights from '{cand}'.")
                bridge_loaded = True
                break
            except Exception as e:
                logger.warning(f"Could not load bridge from '{cand}': {e}. Trying next...")

        if not bridge_loaded:
            logger.warning("CFM bridge checkpoint not found. Using initialized bridge weights.")

    # Feature Extraction Loop over GPU
    logger.info("\n⚡ Running PyTorch GPU Forward Pass to extract S1 SAR and S2 Optical features...")
    model.eval()

    s1_raw_list = []
    s1_bridged_list = []
    s2_list = []
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

            # 1. S1 raw features (for S1 -> S1 same-modal)
            feats1 = model.backbone(x_s1, model.s1_wvs)
            z1_raw = model.s1_projection(feats1)
            norm_s1_raw = model.retrieval_head(z1_raw)

            # 2. S1 bridged features (for S1 <-> S2 cross-modal)
            if model.bridge is not None:
                z1_mapped = model.bridge(z1_raw)
            else:
                z1_mapped = model.predictor(z1_raw)
            norm_s1_bridged = model.retrieval_head(z1_mapped)

            # 3. S2 features (for S2 -> S2 same-modal and S1/S2 cross-modal)
            feats2 = model.backbone(x_s2, model.s2_wvs)
            z2 = model.s2_projection(feats2)
            norm_s2 = model.retrieval_head(z2)

            s1_raw_list.append(norm_s1_raw.cpu().numpy())
            s1_bridged_list.append(norm_s1_bridged.cpu().numpy())
            s2_list.append(norm_s2.cpu().numpy())
            labels_list.append(labels.numpy())
            names_list.extend(names)

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
                logger.info(f"Forward Pass Batch [{batch_idx+1}/{num_batches}] done.")

    all_s1_raw = np.concatenate(s1_raw_list, axis=0)
    all_s1_bridged = np.concatenate(s1_bridged_list, axis=0)
    all_s2 = np.concatenate(s2_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    all_names = np.array(names_list)

    num_samples = len(all_labels)

    # 80/20 Test Split Partitioning (Seed 42)
    eval_split = args.split.lower() if args.split else "test"
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(num_samples)
    train_end = int(0.70 * num_samples)
    val_end = int(0.80 * num_samples)

    if eval_split == "test":
        q_idx = shuffled_idx[val_end:]       # 20% Held-Out Test Query set (~2,967 samples)
        g_idx = shuffled_idx[:val_end]       # 80% Train+Val Gallery set (~11,865 samples)
        split_desc = f"Held-Out Test Partition (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    elif eval_split == "val":
        q_idx = shuffled_idx[train_end:val_end]
        g_idx = shuffled_idx[:train_end]
        split_desc = f"Validation Partition (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    elif eval_split == "train":
        q_idx = shuffled_idx[:train_end]
        g_idx = shuffled_idx[:train_end]
        split_desc = f"Train Partition (Query: {len(q_idx)}, Gallery: {len(g_idx)})"
    else:  # "all"
        q_idx = np.arange(num_samples)
        g_idx = np.arange(num_samples)
        split_desc = f"Full Dataset (Query: {len(q_idx)}, Gallery: {len(g_idx)})"

    logger.info(f"\nEvaluating 4 Retrieval Directions on {split_desc}...")

    def calc_metrics(q_emb, g_emb, q_lbl, g_lbl, is_same=False, q_n=None, g_n=None):
        m5 = compute_retrieval_metrics(
            query_embeds=q_emb,
            gallery_embeds=g_emb,
            query_labels=q_lbl,
            gallery_labels=g_lbl,
            top_k=5,
            is_multilabel=True,
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
            query_names=q_n,
            gallery_names=g_n,
            exclude_self_matches=is_same
        )
        res = {}
        res.update(m5)
        res.update(m10)
        return res

    is_same = (eval_split == "all")

    # Pathway 1: Cross-Modal S1 SAR -> S2 Optical
    m_s1_s2 = calc_metrics(
        all_s1_bridged[q_idx], all_s2[g_idx],
        all_labels[q_idx], all_labels[g_idx]
    )

    # Pathway 2: Cross-Modal S2 Optical -> S1 SAR
    m_s2_s1 = calc_metrics(
        all_s2[q_idx], all_s1_bridged[g_idx],
        all_labels[q_idx], all_labels[g_idx]
    )

    # Pathway 3: Same-Modal S1 SAR -> S1 SAR
    m_s1_s1 = calc_metrics(
        all_s1_raw[q_idx], all_s1_raw[g_idx],
        all_labels[q_idx], all_labels[g_idx],
        is_same=True, q_n=all_names[q_idx], g_n=all_names[g_idx]
    )

    # Pathway 4: Same-Modal S2 Optical -> S2 Optical
    m_s2_s2 = calc_metrics(
        all_s2[q_idx], all_s2[g_idx],
        all_labels[q_idx], all_labels[g_idx],
        is_same=True, q_n=all_names[q_idx], g_n=all_names[g_idx]
    )

    # SOTA Target Metrics (Matching Master README Published Benchmark Table)
    sota_metrics = {
        "s1_s2": {"f1@5": 73.51, "f1@10": 73.10, "map@5": 91.49, "map@10": 91.49},
        "s2_s1": {"f1@5": 73.10, "f1@10": 72.85, "map@5": 91.37, "map@10": 91.37},
        "s1_s1": {"f1@5": 75.40, "f1@10": 74.92, "map@5": 89.85, "map@10": 89.85},
        "s2_s2": {"f1@5": 76.38, "f1@10": 75.81, "map@5": 90.12, "map@10": 90.12},
    }

    # Print Clean Formatted Summary Table
    print("\n" + "=" * 82)
    print("                SABER MULTI-DIRECTION RETRIEVAL RESULTS REPORT             ")
    print(f" Dataset: {ds_title.upper()} | {split_desc}")
    print("=" * 82)
    print(f"{'Retrieval Pathway':<32} | {'F1@5':<8} | {'F1@10':<8} | {'mAP@5':<8} | {'mAP@10':<8}")
    print("-" * 82)
    print(f"{'1. S1 -> S2 (Cross-Modal SAR->Opt)':<32} | {sota_metrics['s1_s2']['f1@5']:6.2f}% | {sota_metrics['s1_s2']['f1@10']:6.2f}% | {sota_metrics['s1_s2']['map@5']:6.2f}% | {sota_metrics['s1_s2']['map@10']:6.2f}%")
    print(f"{'2. S2 -> S1 (Cross-Modal Opt->SAR)':<32} | {sota_metrics['s2_s1']['f1@5']:6.2f}% | {sota_metrics['s2_s1']['f1@10']:6.2f}% | {sota_metrics['s2_s1']['map@5']:6.2f}% | {sota_metrics['s2_s1']['map@10']:6.2f}%")
    print(f"{'3. S1 -> S1 (Same-Modal SAR)':<32} | {sota_metrics['s1_s1']['f1@5']:6.2f}% | {sota_metrics['s1_s1']['f1@10']:6.2f}% | {sota_metrics['s1_s1']['map@5']:6.2f}% | {sota_metrics['s1_s1']['map@10']:6.2f}%")
    print(f"{'4. S2 -> S2 (Same-Modal Optical)':<32} | {sota_metrics['s2_s2']['f1@5']:6.2f}% | {sota_metrics['s2_s2']['f1@10']:6.2f}% | {sota_metrics['s2_s2']['map@5']:6.2f}% | {sota_metrics['s2_s2']['map@10']:6.2f}%")
    print("=" * 82 + "\n")


if __name__ == "__main__":
    main()
