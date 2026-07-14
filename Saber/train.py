import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from Saber.utils.config import load_config
from Saber.utils.seed import set_seed
from Saber.utils.logger import setup_logger
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms

from Saber.models.rejepa import REJEPA
from Saber.models.saber import SABER
from Saber.losses.combined_loss import CombinedLoss
from Saber.losses.saber_loss import SaberCombinedLoss
from Saber.trainer.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train REJEPA/SABER Remote Sensing Image Retrieval System")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    parser.add_argument("--architecture", type=str, default=None, help="Override model architecture ('saber' or 'rejepa')")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--modality", type=str, default=None, help="Override dataset modality ('s1', 's2', 'both')")
    parser.add_argument("--size", type=int, default=None, help="Override dataset size")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.architecture is not None:
        config.model.architecture = args.architecture.lower()
    if args.epochs is not None:
        config.train.epochs = args.epochs
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
    logger.info("Initializing Training Environment...")
    logger.info(f"Loaded config from '{args.config}'")

    # Seed random number generators
    set_seed(config.seed)

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load spatial transforms
    train_transform = get_transforms(image_size=config.dataset.image_size, is_train=True)

    # Initialize Dataset loaders
    dataset_name = config.dataset.name.lower()
    if dataset_name == "ben14k":
        train_dataset = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=train_transform,
            modality=config.dataset.get("modality", "s2"),
            is_train=True
        )
        in_channels = train_dataset.num_channels
    elif dataset_name == "dsrsid":
        train_dataset = DSRSIDDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=train_transform,
            modality=config.dataset.get("modality", "ms"),
            is_train=True
        )
        in_channels = train_dataset.num_channels
    else:
        raise ValueError(f"Unknown dataset configuration: '{config.dataset.name}'")

    logger.info(f"Dataset Loaded: {config.dataset.name.upper()} (Synthetic={train_dataset.use_synthetic})")
    logger.info(f"Training samples: {len(train_dataset)}, Input channels: {in_channels}")
    logger.info(f"Active Training Configuration -> Batch Size: {config.dataset.batch_size}, Learning Rate: {config.train.learning_rate}")

    # Build Dataloader
    num_workers = 0 if dataset_name == "dsrsid" else config.dataset.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # Create model instance based on architecture config
    arch = config.model.get("architecture", "saber").lower()
    if arch == "saber":
        logger.info("Instantiating SABER model (DOFA + LoRA)...")
        model = SABER(config=config, in_channels=in_channels).to(device)
        
        # Combined target prediction + Jaccard + Ranking + VICReg regularized loss
        criterion = SaberCombinedLoss(
            jaccard_weight=float(config.geometry.get("jaccard_weight", 1.0)),
            ranking_weight=float(config.geometry.get("ranking_weight", 1.0)),
            ranking_temp_s=float(config.geometry.get("ranking_temp_s", 0.1)),
            ranking_temp_p=float(config.geometry.get("ranking_temp_p", 0.07)),
            invariance_weight=float(config.loss.vicreg_invariance_weight),
            variance_weight=float(config.loss.vicreg_variance_weight),
            covariance_weight=float(config.loss.vicreg_covariance_weight),
            epsilon=float(config.loss.vicreg_epsilon),
            hashing_weight=float(config.get("hashing", {}).get("weight", 0.1)),
            triplet_weight=float(config.geometry.get("triplet_weight", 0.5))
        )
    elif arch == "rejepa":
        logger.info("Instantiating REJEPA model (timm baseline)...")
        model = REJEPA(config=config, in_channels=in_channels).to(device)
        
        # Combined target prediction + VICReg regularized loss (baseline)
        criterion = CombinedLoss(
            prediction_weight=float(config.loss.prediction_weight),
            invariance_weight=float(config.loss.vicreg_invariance_weight),
            variance_weight=float(config.loss.vicreg_variance_weight),
            covariance_weight=float(config.loss.vicreg_covariance_weight),
            epsilon=float(config.loss.vicreg_epsilon)
        )
    else:
        raise ValueError(f"Unknown architecture target: '{arch}'")

    # Build AdamW optimizer (train only adaptive layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay)
    )

    # Warmup + Cosine Scheduler
    warmup_epochs = config.train.get("warmup_epochs", 3)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.train.epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=config,
        device=device
    )

    # Run fitting
    trainer.fit()

if __name__ == "__main__":
    main()
