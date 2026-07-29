import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from Saber.utils.config import load_config
from Saber.models.saber import SABER

def test_saber_model_forward():
    """Verify SABER forward pass, output shapes, and get_retrieval_embedding method."""
    config = load_config("Saber/configs/config.yaml")
    model = SABER(config=config, in_channels=14)
    model.eval()

    # Bimodal batch (Sentinel-1: 2 channels, Sentinel-2: 12 channels -> total 14)
    dummy_input = torch.randn(2, 14, 120, 120)
    
    with torch.no_grad():
        emb = model.get_retrieval_embedding(dummy_input)
        
    assert emb.shape[0] == 2
    assert emb.shape[1] == 768
    # L2 norm check
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

from Saber.losses.saber_loss import SaberCombinedLoss

def test_saber_loss_dsrsid_and_ben14k():
    """Verify SaberCombinedLoss compatibility with DSRSID (1D) and BEN14K (2D) targets."""
    loss_fn = SaberCombinedLoss()
    
    # 1. DSRSID (8 classes, single label 1D tensor of class indices)
    z1 = torch.randn(4, 768)
    z2 = torch.randn(4, 768)
    z1_pred = torch.randn(4, 768)
    logits_s1_dsrsid = torch.randn(4, 8)
    logits_s2_dsrsid = torch.randn(4, 8)
    targets_dsrsid = torch.tensor([0, 3, 7, 2], dtype=torch.long)
    
    res_dsrsid = loss_fn(
        z1=z1, z2=z2, z1_pred=z1_pred, targets=targets_dsrsid,
        logits_s1=logits_s1_dsrsid, logits_s2=logits_s2_dsrsid
    )
    assert "loss" in res_dsrsid
    assert not torch.isnan(res_dsrsid["loss"])
    
    # 2. BEN-14K (19 classes, multi-label 2D multi-hot tensor)
    logits_s1_ben = torch.randn(4, 19)
    logits_s2_ben = torch.randn(4, 19)
    targets_ben = torch.randint(0, 2, (4, 19)).float()
    
    res_ben = loss_fn(
        z1=z1, z2=z2, z1_pred=z1_pred, targets=targets_ben,
        logits_s1=logits_s1_ben, logits_s2=logits_s2_ben
    )
    assert "loss" in res_ben
    assert not torch.isnan(res_ben["loss"])

if __name__ == "__main__":
    test_saber_model_forward()
    test_saber_loss_dsrsid_and_ben14k()
    print("ALL MODEL FORWARD & LOSS TESTS PASSED CLEANLY!")

