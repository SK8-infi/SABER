import os
import sys
import time
import h5py
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
from Saber.datasets.dsrsid import DSRSID_CLASSES
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import AdvancedFAISSIndex


def extract_real_dsrsid_gallery(mat_path: str, num_samples: int = 1000):
    """
    Extracts num_samples real image pairs from datasets/DSRSID.mat HDF5 file.
    Returns:
        mul_tensors: torch.Tensor of shape (N, 4, 224, 224)
        labels: np.ndarray of shape (N,)
        raw_images: list of PIL Images (RGB visualization of 4-band MS / 1-band PAN)
    """
    logger = setup_logger(name="saber")
    logger.info(f"Loading REAL DSRSID dataset from '{mat_path}' ({num_samples} samples)...")

    with h5py.File(mat_path, "r") as f:
        mul_raw = f["MUL_IMAGES"][:num_samples]       # (N, 4, 64, 64)
        pan_raw = f["PAN_IMAGES"][:num_samples]       # (N, 1, 256, 256)
        labels_raw = f["LAND_COVER_TYPES"][0, :num_samples].astype(int)

    eval_transform = get_transforms(image_size=224, is_train=False)

    mul_tensors = []
    pil_images = []

    for i in range(num_samples):
        img_ms = mul_raw[i].astype(np.float32)  # (4, 64, 64)
        # Normalize
        img_ms = (img_ms - img_ms.min()) / (img_ms.max() - img_ms.min() + 1e-8)
        
        # Transform for model (H, W, C) -> PyTorch tensor
        img_hwc = np.transpose(img_ms, (1, 2, 0)) # (64, 64, 4)
        transformed = eval_transform(image=img_hwc)["image"]  # (4, 224, 224)
        mul_tensors.append(transformed.unsqueeze(0))

        # Create RGB PIL visualization from NIR/Red/Green bands (bands 3, 2, 1)
        rgb = np.stack([img_ms[2], img_ms[1], img_ms[0]], axis=-1)  # (64, 64, 3)
        rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_uint8).resize((224, 224), Image.Resampling.LANCZOS)
        pil_images.append(pil_img)

    mul_tensors = torch.cat(mul_tensors, dim=0)  # (N, 4, 224, 224)
    return mul_tensors, labels_raw, pil_images


def main():
    query_image_path = r"C:\Users\praba\.gemini\antigravity-ide\brain\edf059b8-7681-4446-a6a2-109de032655b\media__1786085808930.jpg"
    mat_path = r"c:\Users\praba\OneDrive\Desktop\LFX26\SABER\datasets\DSRSID.mat"
    checkpoint_path = r"checkpoints/dsrsid/latest.pth"
    artifact_dir = r"C:\Users\praba\.gemini\antigravity-ide\brain\edf059b8-7681-4446-a6a2-109de032655b"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("Saber/configs/config.yaml")

    # 1. Load Checkpoint & Adapt Model Dimensions
    print(f"Loading trained DSRSID model checkpoint: '{checkpoint_path}'...")
    checkpoint_state = load_checkpoint(checkpoint_path, map_location=str(device))
    state_dict = checkpoint_state["model_state_dict"]
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("bridge.") and not k.startswith("classifier.")}

    if "projection_head.fc1.weight" in state_dict:
        config.model.projection_head.hidden_dim = state_dict["projection_head.fc1.weight"].shape[0]
    if "projection_head.fc2.weight" in state_dict:
        config.model.projection_head.out_dim = state_dict["projection_head.fc2.weight"].shape[0]
    if "predictor.predictor.net.0.weight" in state_dict:
        config.model.predictor.hidden_dim = state_dict["predictor.predictor.net.0.weight"].shape[0]

    model = SABER(config=config, in_channels=4).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Successfully loaded SABER DSRSID model!")

    # 2. Load 1,000 REAL DSRSID Images from datasets/DSRSID.mat
    gallery_tensors, gallery_labels, gallery_pil_imgs = extract_real_dsrsid_gallery(mat_path, num_samples=1000)

    # 3. Compute Embeddings for Gallery Images
    print("Extracting 384-D feature embeddings for 1,000 REAL DSRSID scenes...")
    gallery_embeds_list = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(gallery_tensors), batch_size):
            batch_tensors = gallery_tensors[i:i+batch_size].to(device)
            embeds = model.get_retrieval_embedding(batch_tensors)
            gallery_embeds_list.append(embeds.cpu().numpy())
    gallery_embeds = np.concatenate(gallery_embeds_list, axis=0)

    # 4. Build FAISS / Vector Search Index
    index = AdvancedFAISSIndex(
        dimension=gallery_embeds.shape[1],
        metric="cosine",
        index_type="flat"
    )
    index.build_index(gallery_embeds)

    # 5. Process Input Query Satellite Image
    print(f"Processing query image '{query_image_path}'...")
    query_pil = Image.open(query_image_path).convert("RGB")
    eval_transform = get_transforms(image_size=224, is_train=False)

    img_np = np.array(query_pil).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Convert RGB (3 ch) -> 4 ch by adding pseudo-NIR
    if img_tensor.shape[1] == 3:
        nir_ch = img_tensor[:, 0:1, :, :]
        img_tensor = torch.cat([img_tensor, nir_ch], dim=1)

    if img_tensor.shape[-1] != 224 or img_tensor.shape[-2] != 224:
        img_tensor = F.interpolate(img_tensor, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        query_embed = model.get_retrieval_embedding(img_tensor).cpu().numpy()

    # 6. Perform Cosine Similarity Search
    top_k = 5
    similarities, retrieved_indices = index.search(query_embed, k=top_k)

    print("\n" + "=" * 90)
    print(" 🔍 REAL DSRSID IMAGE RETRIEVAL SEARCH RESULTS (datasets/DSRSID.mat)")
    print("=" * 90)
    print(f"{'Rank':<6} | {'Sim Score':<10} | {'Sample Index':<15} | {'DSRSID .mat Location':<30} | {'Class Name'}")
    print("-" * 90)

    retrieved_images_saved = []

    for rank, (idx, sim) in enumerate(zip(retrieved_indices[0], similarities[0]), start=1):
        cls_idx = gallery_labels[idx]
        cls_name = DSRSID_CLASSES[cls_idx] if cls_idx < len(DSRSID_CLASSES) else f"Class {cls_idx}"
        mat_loc = f"datasets/DSRSID.mat -> index #{idx}"

        # Save actual retrieved PIL image patch to artifact folder
        out_img_path = os.path.join(artifact_dir, f"dsrsid_retrieved_rank_{rank}_sample_{idx}.png")
        gallery_pil_imgs[idx].save(out_img_path)
        retrieved_images_saved.append({
            "rank": rank,
            "sim": float(sim),
            "sample_idx": int(idx),
            "class_name": cls_name,
            "mat_location": mat_loc,
            "path": out_img_path
        })

        print(f"#{rank:<5} | {sim*100:6.2f}%    | Sample #{idx:<9} | {mat_loc:<30} | {cls_name}")

    print("=" * 90 + "\n")

    # 7. Render Composite Visual Grid of Real Query vs Real Retrieved Images
    grid_width = 1200
    grid_height = 420
    canvas = Image.new("RGB", (grid_width, grid_height), color=(15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arialbd.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 13)
        font_regular = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = font_bold = font_regular = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (grid_width, 45)], fill=(30, 41, 59))
    draw.text((15, 12), "🛰️ SABER DSRSID REAL SATELLITE IMAGE RETRIEVAL SEARCH (datasets/DSRSID.mat)", fill=(255, 255, 255), font=font_title)

    # Query Box (Left)
    query_resized = query_pil.resize((240, 240), Image.Resampling.LANCZOS)
    canvas.paste(query_resized, (20, 65))
    draw.rectangle([(20, 310), (260, 395)], fill=(30, 41, 59), outline=(56, 189, 248), width=2)
    draw.text((30, 320), "INPUT QUERY SCENE", fill=(56, 189, 248), font=font_bold)
    draw.text((30, 345), "User Attached Scene", fill=(203, 213, 225), font=font_regular)
    draw.text((30, 368), "Agricultural Land Plot", fill=(148, 163, 184), font=font_regular)

    # Top-5 Retrieved Matches (Right)
    for i, item in enumerate(retrieved_images_saved):
        x_pos = 280 + i * 180
        y_pos = 65

        # Paste retrieved image from DSRSID.mat
        ret_pil = Image.open(item["path"]).resize((165, 165), Image.Resampling.LANCZOS)
        canvas.paste(ret_pil, (x_pos, y_pos))

        # Card outline
        draw.rectangle([(x_pos, y_pos), (x_pos + 165, y_pos + 245)], outline=(71, 85, 105), width=1)
        draw.rectangle([(x_pos, y_pos + 165), (x_pos + 165, y_pos + 330)], fill=(30, 41, 59))

        # Badges
        draw.rectangle([(x_pos, y_pos), (x_pos + 45, y_pos + 22)], fill=(15, 23, 42))
        draw.text((x_pos + 8, y_pos + 3), f"#{item['rank']}", fill=(52, 211, 153), font=font_bold)

        draw.text((x_pos + 8, y_pos + 172), f"Sim: {item['sim']*100:.2f}%", fill=(52, 211, 153), font=font_bold)
        draw.text((x_pos + 8, y_pos + 192), f"Class: {item['class_name']}", fill=(255, 255, 255), font=font_bold)
        draw.text((x_pos + 8, y_pos + 212), f"Mat Index: #{item['sample_idx']}", fill=(148, 163, 184), font=font_regular)

    composite_path = os.path.join(artifact_dir, "real_dsrsid_retrieval_visual_grid.png")
    canvas.save(composite_path)
    print(f"Saved real composite retrieval visual grid to '{composite_path}'!")


if __name__ == "__main__":
    main()
