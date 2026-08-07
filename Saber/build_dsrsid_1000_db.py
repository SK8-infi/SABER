import os
import sys
import io
import base64
import h5py
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.utils.checkpoint import load_checkpoint
from Saber.datasets.dsrsid import DSRSID_CLASSES
from Saber.datasets.transforms import get_transforms
from Saber.models.saber import SABER

def build_dsrsid_1000_db():
    mat_path = r"c:\Users\praba\OneDrive\Desktop\LFX26\SABER\datasets\DSRSID.mat"
    checkpoint_path = r"checkpoints/dsrsid/latest.pth"
    out_db_path = r"checkpoints/dsrsid_1000_embeddings.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("Saber/configs/config.yaml")

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

    print(f"Reading 1,000 REAL DSRSID images from '{mat_path}'...")
    num_samples = 1000
    with h5py.File(mat_path, "r") as f:
        mul_raw = f["MUL_IMAGES"][:num_samples]       # (1000, 4, 64, 64)
        labels_raw = f["LAND_COVER_TYPES"][0, :num_samples].astype(int)

    eval_transform = get_transforms(image_size=224, is_train=False)

    mul_tensors = []
    base64_thumbnails = []
    names = []

    for i in range(num_samples):
        img_ms = mul_raw[i].astype(np.float32)
        img_ms = (img_ms - img_ms.min()) / (img_ms.max() - img_ms.min() + 1e-8)

        img_hwc = np.transpose(img_ms, (1, 2, 0))
        transformed = eval_transform(image=img_hwc)["image"]
        mul_tensors.append(transformed.unsqueeze(0))

        # Create RGB PIL visualization (bands 3, 2, 1)
        rgb = np.stack([img_ms[2], img_ms[1], img_ms[0]], axis=-1)
        rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb_uint8).resize((180, 180), Image.Resampling.LANCZOS)

        # Convert PIL -> base64 string
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        base64_thumbnails.append(b64_str)
        names.append(f"DSRSID_sample_{i}.png")

    mul_tensors = torch.cat(mul_tensors, dim=0)

    print("Extracting 384-D feature embeddings...")
    gallery_embeds_list = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(mul_tensors), batch_size):
            batch_tensors = mul_tensors[i:i+batch_size].to(device)
            embeds = model.get_retrieval_embedding(batch_tensors)
            gallery_embeds_list.append(embeds.cpu().numpy())

    gallery_embeds = np.concatenate(gallery_embeds_list, axis=0)

    # Save compact database
    db_data = {
        "num_samples": num_samples,
        "embeddings": gallery_embeds,
        "labels": labels_raw,
        "class_names": DSRSID_CLASSES,
        "names": names,
        "thumbnails": base64_thumbnails
    }

    os.makedirs(os.path.dirname(out_db_path), exist_ok=True)
    torch.save(db_data, out_db_path)
    print(f"Successfully created and saved pre-computed 1,000 DSRSID database to '{out_db_path}'! Size: {os.path.getsize(out_db_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    build_dsrsid_1000_db()
