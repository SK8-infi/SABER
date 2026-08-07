import os
import sys
import argparse
import time
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from Saber.utils.config import load_config
from Saber.utils.logger import setup_logger
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.dsrsid import DSRSIDDataset, DSRSID_CLASSES
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import AdvancedFAISSIndex


def build_dsrsid_gallery(model, device, num_samples: int = 1000):
    logger = setup_logger(name="saber")
    logger.info(f"Building DSRSID Gallery with {num_samples} images using model 'checkpoints/dsrsid/latest.pth'...")

    eval_transform = get_transforms(image_size=224, is_train=False)

    try:
        ds = DSRSIDDataset(
            data_dir="Datasets/DSRSID",
            transform=eval_transform,
            use_synthetic=False,
            modality="ms",
            size=num_samples,
            split="all"
        )
    except Exception:
        logger.warning("Real DSRSID dataset file not found locally. Initializing synthetic benchmark gallery (1,000 images)...")
        ds = DSRSIDDataset(
            data_dir="data",
            transform=eval_transform,
            use_synthetic=True,
            modality="ms",
            size=num_samples,
            split="all"
        )

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)

    model.eval()
    gallery_embeds_list = []
    labels_list = []
    names_list = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch.get("image", batch.get("image1")).to(device)
            if imgs.shape[-1] != 224 or imgs.shape[-2] != 224:
                imgs = F.interpolate(imgs, size=(224, 224), mode="bilinear", align_corners=False)

            labels = batch["label"]
            names = batch["name"]

            # Extract 768-D L2-normalized embeddings via model
            embeds = model.get_retrieval_embedding(imgs)
            gallery_embeds_list.append(embeds.cpu().numpy())
            labels_list.append(labels.numpy())
            names_list.extend(names)

    gallery_embeds = np.concatenate(gallery_embeds_list, axis=0)
    gallery_labels = np.concatenate(labels_list, axis=0)
    gallery_names = np.array(names_list)

    # Build FAISS index
    index = AdvancedFAISSIndex(
        dimension=gallery_embeds.shape[1],
        metric="cosine",
        index_type="flat"
    )
    index.build_index(gallery_embeds)

    return index, gallery_embeds, gallery_labels, gallery_names


def search_image(query_image_path: str, model, index, gallery_names, gallery_labels, device, top_k: int = 5):
    logger = setup_logger(name="saber")
    logger.info(f"Processing query image: '{query_image_path}'...")

    if not os.path.exists(query_image_path):
        raise FileNotFoundError(f"Query image path '{query_image_path}' does not exist!")

    # Open image using PIL
    img_pil = Image.open(query_image_path).convert("RGB")
    eval_transform = get_transforms(image_size=224, is_train=False)

    # Transform to tensor
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad to 4 channels if necessary (Gaofen-1 MS has 4 channels)
    if img_tensor.shape[1] == 3:
        nir_channel = img_tensor[:, 0:1, :, :]  # Use red channel as pseudo-NIR
        img_tensor = torch.cat([img_tensor, nir_channel], dim=1)

    if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    model.eval()
    with torch.no_grad():
        query_embed = model.get_retrieval_embedding(img_tensor).cpu().numpy()

    # FAISS search
    start_time = time.time()
    similarities, retrieved_indices = index.search(query_embed, k=top_k)
    search_time_ms = (time.time() - start_time) * 1000.0

    print("\n" + "=" * 80)
    print(f" 🔍 SABER DSRSID IMAGE RETRIEVAL SEARCH RESULTS (Top-{top_k})")
    print(f" Query Image : '{query_image_path}' | Search Time: {search_time_ms:.2f} ms")
    print("=" * 80)
    print(f"{'Rank':<6} | {'Sim Score':<10} | {'Gallery Image Name':<40} | {'Class Label'}")
    print("-" * 80)

    for rank, (idx, sim) in enumerate(zip(retrieved_indices[0], similarities[0]), start=1):
        g_name = gallery_names[idx]
        lbl = gallery_labels[idx]
        if isinstance(lbl, (np.ndarray, list)) and len(lbl) > 1:
            cls_idx = np.argmax(lbl)
            cls_name = DSRSID_CLASSES[cls_idx] if cls_idx < len(DSRSID_CLASSES) else f"Class {cls_idx}"
        else:
            cls_name = DSRSID_CLASSES[int(lbl)] if int(lbl) < len(DSRSID_CLASSES) else f"Class {int(lbl)}"

        print(f"#{rank:<5} | {sim*100:6.2f}%    | {g_name:<40} | {cls_name}")

    print("=" * 80 + "\n")
    return retrieved_indices[0], similarities[0]


def main():
    parser = argparse.ArgumentParser(description="Search Query Image on 1,000 DSRSID Gallery Images using latest.pth")
    parser.add_argument("--query_image", type=str, required=True, help="Path to input query image (e.g. query.png)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dsrsid/latest.pth", help="Path to trained DSRSID model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top retrieved images to return")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("Saber/configs/config.yaml")

    state_dict = None
    if os.path.exists(args.checkpoint):
        print(f"Loading trained DSRSID model checkpoint: '{args.checkpoint}'...")
        checkpoint_state = load_checkpoint(args.checkpoint, map_location=str(device))
        state_dict = checkpoint_state["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.") and not k.startswith("classifier.")}

        # Auto-adapt config to matching checkpoint dimensions
        if "projection_head.fc1.weight" in state_dict:
            config.model.projection_head.hidden_dim = state_dict["projection_head.fc1.weight"].shape[0]
        if "projection_head.fc2.weight" in state_dict:
            config.model.projection_head.out_dim = state_dict["projection_head.fc2.weight"].shape[0]
        if "predictor.predictor.net.0.weight" in state_dict:
            config.model.predictor.hidden_dim = state_dict["predictor.predictor.net.0.weight"].shape[0]

    model = SABER(config=config, in_channels=4).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded model checkpoint!")
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Using initialized backbone.")

    # 1. Build 1,000-image DSRSID gallery
    index, gallery_embeds, gallery_labels, gallery_names = build_dsrsid_gallery(model, device, num_samples=1000)

    # 2. Perform retrieval search on input query image
    search_image(args.query_image, model, index, gallery_names, gallery_labels, device, top_k=args.top_k)


if __name__ == "__main__":
    main()
