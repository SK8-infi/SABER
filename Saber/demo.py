import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from Saber.utils.config import load_config
from Saber.utils.checkpoint import load_checkpoint
from datasets.ben14k import BEN14KDataset
from datasets.dsrsid import DSRSIDDataset
from datasets.transforms import get_transforms
from Saber.models.rejepa import REJEPA
from Saber.retrieval.faiss_index import FAISSIndex
from Saber.retrieval.retriever import Retriever
from Saber.visualization.similarity import plot_retrieval_grid
from Saber.visualization.attention import visualize_attention_map

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the FAISS index using a query image")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pth", help="Path to model checkpoint")
    parser.add_argument("--query_index", type=int, default=0, help="Index of sample in validation set to use as query")
    parser.add_argument("--query_path", type=str, default=None, help="Path to external query image file")
    parser.add_argument("--synthetic", type=str, default=None, help="Force synthetic dataset mode ('true' or 'false')")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # CLI Overrides
    if args.synthetic is not None:
        config.dataset.use_synthetic = (args.synthetic.lower() == "true")

    # Set up Logger
    from Saber.utils.logger import setup_logger
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    logger.info("Initializing REJEPA Retrieval Query Demo...")

    # Establish target device
    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info(f"Computation Device: {device}")

    # Load transform
    eval_transform = get_transforms(image_size=config.dataset.image_size, is_train=False)

    # Initialize Dataset (to select query images or load external ones)
    dataset_name = config.dataset.name.lower()
    if dataset_name == "ben14k":
        dataset = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            image_size=config.dataset.image_size,
            transform=eval_transform,
            modality=config.dataset.get("modality", "s2"),
            is_train=False
        )
        in_channels = dataset.num_channels
        is_multilabel = True
    elif dataset_name == "dsrsid":
        dataset = DSRSIDDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            image_size=config.dataset.image_size,
            transform=eval_transform,
            is_train=False
        )
        in_channels = dataset.num_channels
        is_multilabel = False
    else:
        raise ValueError(f"Unknown dataset configuration: '{config.dataset.name}'")

    # Load FAISS index database
    faiss_index = FAISSIndex(
        dimension=config.model.projection_head.out_dim,
        metric=config.retrieval.metric
    )
    if not os.path.exists(config.retrieval.index_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{config.retrieval.index_path}'. "
            f"Please run evaluation/indexing first: python Saber/evaluate.py"
        )
    faiss_index.load_index(config.retrieval.index_path)

    # Load gallery metadata sidecar
    metadata_path = os.path.splitext(config.retrieval.index_path)[0] + "_metadata.pth"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Gallery metadata sidecar not found at '{metadata_path}'. "
            f"Please run evaluation/indexing first: python Saber/evaluate.py"
        )
    try:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except TypeError:
        metadata = torch.load(metadata_path, map_location="cpu")
    gallery_names = metadata["names"]
    gallery_labels = metadata["labels"]
    logger.info(f"Loaded gallery metadata containing {len(gallery_names)} records.")

    # Instantiate Retriever
    retriever = Retriever(
        index=faiss_index,
        gallery_names=gallery_names,
        gallery_labels=gallery_labels
    )

    # Create REJEPA model instance
    model = REJEPA(config=config, in_channels=in_channels).to(device)

    # Load model checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            logger.info(f"Loading checkpoint parameters from: '{args.checkpoint}'")
            checkpoint_state = load_checkpoint(args.checkpoint, map_location=str(device))
            model.load_state_dict(checkpoint_state["model_state_dict"])
            logger.info("Successfully loaded model parameters.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Running with initialized model weights.")
    else:
        logger.warning("No model checkpoint loaded. Running query with initialized model weights.")

    # Determine query image
    query_name = ""
    query_gt_label = None

    if args.query_path and os.path.exists(args.query_path):
        logger.info(f"Loading external query image: '{args.query_path}'")
        # Standard PIL load and transform
        from PIL import Image
        pil_img = Image.open(args.query_path)
        img_np = np.array(pil_img)
        # Handle channels
        if len(img_np.shape) == 2:
            img_np = np.expand_dims(img_np, axis=-1)
        if img_np.shape[-1] != in_channels:
            # Replicate or pad
            if img_np.shape[-1] > in_channels:
                img_np = img_np[..., :in_channels]
            else:
                pad = np.zeros((img_np.shape[0], img_np.shape[1], in_channels - img_np.shape[-1]), dtype=img_np.dtype)
                img_np = np.concatenate([img_np, pad], axis=-1)
        
        # Apply eval transform
        augmented = eval_transform(image=img_np.astype(np.float32))
        query_tensor = augmented["image"]
        query_name = os.path.basename(args.query_path)
    else:
        # Load from dataset
        logger.info(f"Fetching query sample at dataset index {args.query_index}...")
        sample = dataset[args.query_index]
        query_tensor = sample["image"]
        query_gt_label = sample["label"].numpy()
        query_name = sample["name"]

    # Extract query embedding
    model.eval()
    with torch.no_grad():
        query_img_batch = query_tensor.unsqueeze(0).to(device)
        query_embedding = model.get_retrieval_embedding(query_img_batch).cpu().numpy()[0]

    # Perform nearest neighbor retrieval
    top_k = config.retrieval.top_k
    logger.info(f"Retrieving top {top_k} nearest matches from FAISS...")
    matches = retriever.retrieve(query_embedding, k=top_k)

    # Print results to stdout
    logger.info("=========================================")
    logger.info(f"QUERY IMAGE: {query_name}")
    if query_gt_label is not None:
        if is_multilabel:
            logger.info(f"Ground Truth Labels (Indices): {np.where(query_gt_label > 0)[0]}")
        else:
            logger.info(f"Ground Truth Class Index    : {query_gt_label}")
    logger.info("=========================================")
    logger.info("RETRIEVED MATCHES:")
    for rank, match in enumerate(matches, 1):
        logger.info(f"Rank {rank}: {match['name']}")
        logger.info(f"  Similarity Score: {match['score']:.4f}")
        if is_multilabel:
            logger.info(f"  Active Classes  : {np.where(match['label'] > 0)[0]}")
        else:
            logger.info(f"  Class Index     : {match['label']}")
    logger.info("=========================================")

    # Load matching images from gallery for grid visualization
    logger.info("Preparing visual retrieval grid...")
    retrieved_imgs = []
    retrieved_names = []
    retrieved_scores = []

    # Prepare raw query image for grid (move dimensions back if necessary)
    raw_query_img = query_tensor.permute(1, 2, 0).numpy()

    # Loop matching metadata to fetch images
    for match in matches:
        match_name = match["name"]
        match_score = match["score"]
        
        img_arr = None
        if not dataset.use_synthetic:
            # Load real image from dataset using index parsing/matching
            if config.dataset.name.lower() == "dsrsid":
                try:
                    parts = match_name.split("_")
                    idx_str = parts[-1].split(".")[0]
                    match_idx = int(idx_str)
                    
                    old_train = dataset.is_train
                    dataset.is_train = False
                    item = dataset[match_idx]
                    dataset.is_train = old_train
                    
                    img_tensor = item["image"]
                    img_arr = img_tensor.permute(1, 2, 0).numpy()
                except Exception as e:
                    logger.error(f"Error loading real DSRSID image for display: {e}")
            elif config.dataset.name.lower() == "ben14k":
                try:
                    match_id = match_name.replace(".png", "").replace("_paired", "")
                    if dataset.modality == "s1":
                        mask = dataset.df["S1_ID"] == match_id
                    else:
                        mask = dataset.df["S2_ID"] == match_id
                        
                    indices = dataset.df.index[mask].tolist()
                    if indices:
                        match_idx = indices[0]
                        old_train = dataset.is_train
                        dataset.is_train = False
                        item = dataset[match_idx]
                        dataset.is_train = old_train
                        
                        img_tensor = item["image"]
                        img_arr = img_tensor.permute(1, 2, 0).numpy()
                except Exception as e:
                    logger.error(f"Error loading real BEN-14K image for display: {e}")
        
        if img_arr is None:
            # Fallback to generating a synthetic tensor representing the match
            dummy_match = np.random.randn(config.dataset.image_size, config.dataset.image_size, in_channels).astype(np.float32)
            retrieved_imgs.append(dummy_match)
        else:
            retrieved_imgs.append(img_arr)
            
        retrieved_names.append(match_name)
        retrieved_scores.append(match_score)

    # Save visual grid
    os.makedirs(config.viz_dir, exist_ok=True)
    grid_path = os.path.join(config.viz_dir, "retrieval_results.png")
    plot_retrieval_grid(
        query_img=raw_query_img,
        retrieved_imgs=retrieved_imgs,
        scores=retrieved_scores,
        names=retrieved_names,
        save_path=grid_path
    )
    logger.info(f"Retrieval grid plot saved to: {grid_path}")

    # Generate Attention Map
    attention_path = os.path.join(config.viz_dir, "query_attention.png")
    try:
        logger.info("Generating self-attention map overlay for query...")
        visualize_attention_map(
            model=model,
            image_tensor=query_tensor,
            save_path=attention_path
        )
        logger.info(f"Query attention map overlay saved to: {attention_path}")
    except Exception as e:
        logger.warning(f"Could not generate attention map overlay: {e}")

    logger.info("Demo execution completed successfully.")

if __name__ == "__main__":
    main()
