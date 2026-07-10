import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.rejepa import REJEPA

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract S1 and S2 latent features from trained REJEPA")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crossmodal/latest.pth", help="Path to trained model checkpoint file (.pth)")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="Saber/extracted", help="Directory to save extracted features")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.synthetic is not None:
        config.dataset.use_synthetic = (args.synthetic.lower() == "true")
    if args.dataset_name is not None:
        config.dataset.name = args.dataset_name
    if args.data_dir is not None:
        config.dataset.data_dir = args.data_dir

    # Set config modality to "both" for cross-modal feature extraction
    config.dataset.modality = "both"

    # Set up Logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("Initializing Feature Extraction runner...")

    # Seed random number generators
    set_seed(config.seed)

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load spatial transforms (is_train=False)
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)

    # Initialize Dataset loader (is_train=False)
    dataset_name = config.dataset.name.lower()
    if dataset_name == "ben14k":
        eval_dataset = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=eval_transform,
            modality="both",
            is_train=False
        )
        in_channels = eval_dataset.num_channels
    else:
        raise ValueError(f"Feature extraction only supported for bimodal BEN-14K currently. Dataset is '{config.dataset.name}'")

    logger.info(f"Dataset Loaded: {config.dataset.name.upper()} (Synthetic={eval_dataset.use_synthetic})")
    logger.info(f"Total samples: {len(eval_dataset)}")

    # Build Dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers
    )

    # Create REJEPA model instance
    model = REJEPA(config=config, in_channels=in_channels).to(device)

    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading checkpoint parameters from: '{args.checkpoint}'")
        checkpoint_state = load_checkpoint(args.checkpoint, map_location=str(device))
        model.load_state_dict(checkpoint_state["model_state_dict"])
        logger.info("Successfully loaded model parameters.")
    else:
        logger.error(f"Checkpoint not found at: '{args.checkpoint}'")
        sys.exit(1)

    model.eval()

    s1_feats_list = []
    s2_feats_list = []
    labels_list = []
    names_list = []

    logger.info("Extracting projection latents...")
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            img_s1 = batch["image_s1"].to(device)
            img_s2 = batch["image_s2"].to(device)
            labels = batch["label"]
            names = batch["name"]

            # Compute projection head outputs
            feats_s1 = model.backbone(model.adapter_s1(img_s1))
            z_s1 = model.projection_head(feats_s1)

            feats_s2 = model.backbone(model.adapter_s2(img_s2))
            z_s2 = model.projection_head(feats_s2)

            s1_feats_list.append(z_s1.cpu().numpy())
            s2_feats_list.append(z_s2.cpu().numpy())
            labels_list.append(labels.numpy())
            names_list.extend(names)

    # Concatenate results
    s1_feats = np.concatenate(s1_feats_list, axis=0)
    s2_feats = np.concatenate(s2_feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save to disk
    np.save(os.path.join(args.output_dir, "s1_feats.npy"), s1_feats)
    np.save(os.path.join(args.output_dir, "s2_feats.npy"), s2_feats)
    np.save(os.path.join(args.output_dir, "labels.npy"), labels)
    np.save(os.path.join(args.output_dir, "names.npy"), np.array(names_list))

    logger.info(f"Successfully saved features to: '{args.output_dir}'")
    logger.info(f"s1_feats shape: {s1_feats.shape}")
    logger.info(f"s2_feats shape: {s2_feats.shape}")
    logger.info(f"labels shape: {labels.shape}")

if __name__ == "__main__":
    main()
