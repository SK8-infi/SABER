import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Saber_bridge.utils.config import load_config
from Saber_bridge.utils.seed import set_seed
from Saber_bridge.utils.logger import setup_logger
from Saber_bridge.utils.checkpoint import load_checkpoint
from datasets.ben14k import BEN14KDataset
from datasets.transforms import get_transforms
from Saber_bridge.models.rejepa import REJEPA
from Saber_bridge.models.bridge import CFMBridge
from Saber_bridge.trainer.evaluator import Evaluator
from Saber_bridge.retrieval.faiss_index import FAISSIndex
from Saber_bridge.visualization.tsne import plot_tsne
from Saber_bridge.visualization.umap import plot_umap

class CFMBridgeWrapper(nn.Module):
    def __init__(self, cfm_bridge: nn.Module, ode_steps: int = 5) -> None:
        super().__init__()
        self.cfm_bridge = cfm_bridge
        self.ode_steps = ode_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the S1 query projection. We integrate ODE to map S1 -> S2.
        z = x.clone()
        device = x.device
        if self.ode_steps == 1:
            tau = torch.zeros(z.shape[0], 1, device=device)
            v, _ = self.cfm_bridge(z, tau, x)
            z = z + v
        else:
            dt = 1.0 / self.ode_steps
            for step in range(self.ode_steps):
                tau = torch.ones(z.shape[0], 1, device=device) * (step * dt)
                v, _ = self.cfm_bridge(z, tau, x)
                z = z + v * dt
        return z

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate REJEPA with CFM Latent Bridge")
    parser.add_argument("--config", type=str, default="Saber_bridge/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/crossmodal/latest.pth", help="Path to trained model checkpoint file (.pth)")
    parser.add_argument("--bridge_checkpoint", type=str, default="Saber_bridge/checkpoints/bridge_best.pth", help="Path to trained bridge checkpoint file (.pth)")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--ode_steps", type=int, default=5, help="Number of ODE solver integration steps")
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

    # Force bimodal modality
    config.dataset.modality = "both"

    # Set up Logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info(f"Initializing REJEPA Evaluation with CFM Latent Bridge ({args.ode_steps} steps)...")

    # Seed random number generators
    set_seed(config.seed)

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load val/test spatial transforms (is_train=False)
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
        raise ValueError("Only BEN-14K supported currently for bimodal cross-modal bridge evaluation.")

    logger.info(f"Dataset Loaded: {config.dataset.name.upper()} (Synthetic={eval_dataset.use_synthetic})")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Build Dataloader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers
    )

    # Create REJEPA model instance
    model = REJEPA(config=config, in_channels=in_channels).to(device)

    # Load REJEPA encoder checkpoints
    if args.checkpoint and os.path.exists(args.checkpoint):
        logger.info(f"Loading REJEPA checkpoint parameters from: '{args.checkpoint}'")
        checkpoint_state = load_checkpoint(args.checkpoint, map_location=str(device))
        model.load_state_dict(checkpoint_state["model_state_dict"])
        logger.info("Successfully loaded REJEPA model parameters.")
    else:
        logger.warning(f"REJEPA Checkpoint not found at: '{args.checkpoint}'")

    # Load trained CFM Bridge
    bridge = CFMBridge(dim=384, hidden_dim=512, num_blocks=3).to(device)
    if args.bridge_checkpoint and os.path.exists(args.bridge_checkpoint):
        logger.info(f"Loading trained CFM Bridge checkpoint from: '{args.bridge_checkpoint}'")
        bridge.load_state_dict(torch.load(args.bridge_checkpoint, map_location=str(device)))
        logger.info("Successfully loaded bridge model parameters.")
    else:
        logger.error(f"Bridge Checkpoint not found at: '{args.bridge_checkpoint}'. Cannot proceed.")
        sys.exit(1)

    # Wrap the CFM bridge with ODE solver and override model predictor dynamically
    model.predictor = CFMBridgeWrapper(bridge, ode_steps=args.ode_steps)
    model.eval()

    # Initialize Evaluator
    evaluator = Evaluator(
        model=model,
        dataloader=eval_loader,
        device=device,
        config=config
    )

    # Run evaluation
    results = evaluator.evaluate(top_k=config.retrieval.top_k)

    # Log metrics
    logger.info("=========================================")
    logger.info("   RETRIEVAL METRICS (WITH CFM BRIDGE)   ")
    logger.info("=========================================")
    for metric_name, val in results["metrics"].items():
        logger.info(f"{metric_name.upper():<15}: {val:.4f}")
    logger.info("=========================================")

    # Get Gallery and Query representations
    all_embeddings = results["embeddings"]
    gallery_indices = results["gallery_indices"]
    
    gallery_embeddings = all_embeddings[gallery_indices]
    gallery_labels = results["labels"][gallery_indices]
    gallery_names = [results["names"][i] for i in gallery_indices]

    # Save outputs to different folder to not overwrite baseline
    output_dir = "Saber_bridge/checkpoints_bridge"
    os.makedirs(output_dir, exist_ok=True)

    # Build and serialize the FAISS index (using gallery items)
    faiss_index = FAISSIndex(
        dimension=config.model.projection_head.out_dim,
        metric=config.retrieval.metric
    )
    faiss_index.build_index(gallery_embeddings)
    
    index_path = os.path.join(output_dir, "faiss_index_bridge.bin")
    faiss_index.save_index(index_path)
    logger.info(f"Saved FAISS Index to: {index_path}")
    
    metadata_path = os.path.join(output_dir, "faiss_index_bridge_metadata.pth")
    torch.save({
        "names": gallery_names,
        "labels": gallery_labels,
        "embeddings": gallery_embeddings,
        "dataset_name": config.dataset.name,
        "modality": getattr(eval_dataset, "modality", "both")
    }, metadata_path)
    logger.info(f"Saved gallery metadata to: {metadata_path}")

    # Generate and save projections
    viz_dir = "Saber_bridge/visualizations_bridge"
    os.makedirs(viz_dir, exist_ok=True)
    
    tsne_path = os.path.join(viz_dir, "tsne.png")
    plot_tsne(
        embeddings=all_embeddings,
        labels=results["labels"],
        save_path=tsne_path,
        perplexity=config.visualization.tsne_perplexity,
        n_iter=config.visualization.tsne_n_iter
    )
    logger.info(f"Saved t-SNE plot to: {tsne_path}")

    umap_path = os.path.join(viz_dir, "umap.png")
    plot_umap(
        embeddings=all_embeddings,
        labels=results["labels"],
        save_path=umap_path,
        n_neighbors=config.visualization.umap_n_neighbors,
        min_dist=config.visualization.umap_min_dist
    )
    logger.info(f"Saved UMAP plot to: {umap_path}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
