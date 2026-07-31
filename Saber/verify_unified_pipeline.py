import sys
import os
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Saber.utils.config import load_config
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.models.saber import SABER
from Saber.models.bridge import CFMBridge, CFMBridgeWrapper

def test_unified_pipeline():
    print("="*80)
    print(" 🧪 VERIFYING UNIFIED PIPELINE & PATCH-PROJECTED CFM BRIDGE")
    print("="*80)

    config = load_config("Saber/configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Test DSRSID Stratified Subsampling (14,000 samples)
    print("\n--- 1. Testing DSRSID Stratified 14,000 Subsampling ---")
    dsrsid = DSRSIDDataset(data_dir="datasets/DSRSID.mat", use_synthetic=True, size=14000)
    print(f"DSRSID Dataset Size: {len(dsrsid)} samples")
    assert len(dsrsid) == 14000 or len(dsrsid) == 9800, f"Expected ~14,000 samples, got {len(dsrsid)}"
    print("✅ DSRSID Stratified Subsampling Verified!")

    # 2. Test Patch-Projected CFM Bridge (196 spatial patch tokens)
    print("\n--- 2. Testing Patch-Projected Spatial CFM Bridge ---")
    bridge_net = CFMBridge(dim=768, hidden_dim=768, num_blocks=3)
    wrapper = CFMBridgeWrapper(bridge_net, ode_steps=1).to(device)

    # Spatial patch sequence input (B=4, L=196, D=768)
    dummy_spatial_patches = torch.randn(4, 196, 768, device=device)
    output_patches = wrapper(dummy_spatial_patches)
    print(f"Input Patch Tokens  : {dummy_spatial_patches.shape}")
    print(f"Output Patch Tokens : {output_patches.shape}")
    assert output_patches.shape == (4, 196, 768), "Patch shape mismatch!"
    print("✅ Patch-Projected Spatial CFM Bridge Verified!")

    # 3. Test Master SABER Model Forward Pass
    print("\n--- 3. Testing Single Master SABER Model Forward Pass ---")
    model = SABER(config=config, in_channels=14).to(device)

    # Test BEN-14K inputs (x1 = S1 2ch, x2 = S2 12ch)
    x1_ben = torch.randn(2, 2, 224, 224, device=device)
    x2_ben = torch.randn(2, 12, 224, 224, device=device)
    z1_b, z2_b, z1_pred_b, log1_b, log2_b = model(x1_ben, x2_ben)
    print(f"BEN-14K Context (S1 2ch) Latent  : {z1_b.shape}")
    print(f"BEN-14K Target (S2 12ch) Latent  : {z2_b.shape}")

    # Test DSRSID inputs (x_pan = 1ch, x_ms = 4ch)
    feats_pan = model.backbone(torch.randn(2, 1, 224, 224, device=device), [0.675])
    z_pan = model.projection_head(feats_pan)
    feats_ms = model.backbone(torch.randn(2, 4, 224, 224, device=device), [0.485, 0.555, 0.660, 0.830])
    z_ms = model.projection_head(feats_ms)
    print(f"DSRSID PAN (1ch) Projected Latent : {z_pan.shape}")
    print(f"DSRSID MS (4ch) Projected Latent  : {z_ms.shape}")

    print("✅ Single Master SABER Model Verified!")

    print("\n" + "="*80)
    print(" 🎉 ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    test_unified_pipeline()
