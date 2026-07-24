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

if __name__ == "__main__":
    test_saber_model_forward()
    print("ALL MODEL FORWARD TESTS PASSED CLEANLY!")
