import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from rejepa.utils.config import load_config
from rejepa.utils.seed import set_seed
from rejepa.utils.logger import setup_logger
from rejepa.utils.checkpoint import load_checkpoint
from datasets.ben14k import BEN14KDataset
from datasets.dsrsid import DSRSIDDataset
from datasets.transforms import get_transforms
from rejepa.models.rejepa import REJEPA
from rejepa.trainer.evaluator import Evaluator
from rejepa.retrieval.faiss_index import FAISSIndex
from rejepa.visualization.tsne import plot_tsne
from rejepa.visualization.umap import plot_umap
from rejepa.visualization.similarity import plot_similarity_matrix

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate REJEPA Retrieval performance and build FAISS Index")
    parser.add_argument("--config", type=str, default="rejepa/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint file (.pth)")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name ('ben14k' or 'dsrsid')")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to dataset directory")
    parser.add_argument("--modality", type=str, default=None, help="Override dataset modality ('s1', 's2', 'both')")
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
    if args.modality is not None:
        config.dataset.modality = args.modality

    # Set up Logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("Initializing REJEPA Evaluation & Indexing runner...")

    # Seed random number generators
    set_seed(config.seed)

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load val/test spatial transforms (is_train=False)
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)

    # Initialize Dataset loaders (is_train=False)
    dataset_name = config.dataset.name.lower()
    if dataset_name == "ben14k":
        eval_dataset = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=eval_transform,
            modality=config.dataset.get("modality", "s2"),
            is_train=False
        )
        in_channels = eval_dataset.num_channels
    elif dataset_name == "dsrsid":
        eval_dataset = DSRSIDDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=eval_transform,
            is_train=False
        )
        in_channels = eval_dataset.num_channels
    else:
        raise ValueError(f"Unknown dataset configuration: '{config.dataset.name}'")

    logger.info(f"Dataset Loaded: {config.dataset.name.upper()} (Synthetic={eval_dataset.use_synthetic})")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")

    # Build Dataloader
    num_workers = 0 if dataset_name == "dsrsid" else config.dataset.num_workers
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Create REJEPA model instance
    model = REJEPA(config=config, in_channels=in_channels).to(device)

    # Load checkpoint parameters if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            logger.info(f"Loading checkpoint parameters from: '{args.checkpoint}'")
            checkpoint_state = load_checkpoint(args.checkpoint, map_location=str(device))
            model.load_state_dict(checkpoint_state["model_state_dict"])
            logger.info("Successfully loaded model parameters.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Proceeding with initialized weights.")
    else:
        logger.warning("No valid model checkpoint specified or found. Running evaluation with initialized model weights.")

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
    logger.info("           RETRIEVAL METRICS             ")
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

    # Build and serialize the FAISS index (using gallery items)
    faiss_index = FAISSIndex(
        dimension=config.model.projection_head.out_dim,
        metric=config.retrieval.metric
    )
    faiss_index.build_index(gallery_embeddings)
    
    # Save FAISS Index
    faiss_index.save_index(config.retrieval.index_path)

    # Save gallery metadata sidecar (for fast labels/filenames matching in demo queries)
    metadata_path = os.path.splitext(config.retrieval.index_path)[0] + "_metadata.pth"
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
        
    torch.save({
        "names": gallery_names,
        "labels": gallery_labels,
        "embeddings": gallery_embeddings,
        "dataset_name": config.dataset.name,
        "modality": getattr(eval_dataset, "modality", "s2")
    }, metadata_path)
    logger.info(f"Saved gallery metadata to: {metadata_path}")

    # Generate and save projections and similarity heatmaps
    logger.info("Generating visualization plots...")
    os.makedirs(config.viz_dir, exist_ok=True)
    
    tsne_path = os.path.join(config.viz_dir, "tsne.png")
    plot_tsne(
        embeddings=all_embeddings,
        labels=results["labels"],
        save_path=tsne_path,
        perplexity=config.visualization.tsne_perplexity,
        n_iter=config.visualization.tsne_n_iter
    )
    logger.info(f"Saved t-SNE plot to: {tsne_path}")

    umap_path = os.path.join(config.viz_dir, "umap.png")
    plot_umap(
        embeddings=all_embeddings,
        labels=results["labels"],
        save_path=umap_path,
        n_neighbors=config.visualization.umap_n_neighbors,
        min_dist=config.visualization.umap_min_dist
    )
    logger.info(f"Saved UMAP plot to: {umap_path}")

    if results.get("similarity_matrix") is not None:
        sim_path = os.path.join(config.viz_dir, "similarity_heatmap.png")
        plot_similarity_matrix(
            similarity_matrix=results["similarity_matrix"],
            save_path=sim_path
        )
        logger.info(f"Saved similarity matrix heatmap to: {sim_path}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()
