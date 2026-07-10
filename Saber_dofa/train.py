import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader

from Saber_dofa.utils.config import load_config
from Saber_dofa.utils.seed import set_seed
from Saber_dofa.utils.logger import setup_logger
from Saber_dofa.datasets.ben14k import BEN14KDataset
from Saber_dofa.datasets.dsrsid import DSRSIDDataset
from Saber_dofa.datasets.transforms import get_transforms
from Saber_dofa.models.rejepa import REJEPA
from Saber_dofa.models.saber import SABER
from Saber_dofa.losses.combined_loss import CombinedLoss
from Saber_dofa.trainer.trainer import Trainer

def main() -> None:
    parser = argparse.ArgumentParser(description="Train REJEPA-style Remote Sensing Image Retrieval System")
    parser.add_argument("--config", type=str, default="Saber_dofa/configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--modality", type=str, default=None, help="Override dataset modality ('s1', 's2', 'both')")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
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

    # Set up Logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("Initializing REJEPA Training Environment...")
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
            is_train=True
        )
        in_channels = train_dataset.num_channels
    else:
        raise ValueError(f"Unknown dataset configuration: '{config.dataset.name}'")

    logger.info(f"Dataset Loaded: {config.dataset.name.upper()} (Synthetic={train_dataset.use_synthetic})")
    logger.info(f"Training samples: {len(train_dataset)}, Input channels: {in_channels}")

    # Build Dataloader
    num_workers = 0 if dataset_name == "dsrsid" else config.dataset.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    # Create model instance based on architecture config
    arch = config.model.get("architecture", "rejepa")
    if arch == "saber":
        logger.info("Instantiating SABER model (DOFA + LoRA)...")
        model = SABER(config=config, in_channels=in_channels).to(device)
    else:
        logger.info("Instantiating REJEPA baseline model...")
        model = REJEPA(config=config, in_channels=in_channels).to(device)

    # Build AdamW optimizer (train only adaptive layers)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay)
    )

    # Cosine Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs,
        eta_min=1e-6
    )

    # Combined target prediction + VICReg regularized loss
    criterion = CombinedLoss(
        prediction_weight=float(config.loss.prediction_weight),
        invariance_weight=float(config.loss.vicreg_invariance_weight),
        variance_weight=float(config.loss.vicreg_variance_weight),
        covariance_weight=float(config.loss.vicreg_covariance_weight),
        epsilon=float(config.loss.vicreg_epsilon)
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
