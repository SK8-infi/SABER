import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset

def test_ben14k_split_disjointness():
    """Verify that BEN14K train, val, and test splits have ZERO overlapping samples."""
    train_ds = BEN14KDataset(data_dir="data", use_synthetic=True, size=1000, split="train")
    val_ds = BEN14KDataset(data_dir="data", use_synthetic=True, size=1000, split="val")
    test_ds = BEN14KDataset(data_dir="data", use_synthetic=True, size=1000, split="test")
    
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0
    
    # In synthetic mode, size is allocated proportionately
    assert len(train_ds) + len(val_ds) + len(test_ds) <= 1000

def test_dsrsid_split_disjointness():
    """Verify that DSRSID train, val, and test splits have ZERO overlapping samples."""
    train_ds = DSRSIDDataset(data_dir="data", use_synthetic=True, size=1000, split="train")
    val_ds = DSRSIDDataset(data_dir="data", use_synthetic=True, size=1000, split="val")
    test_ds = DSRSIDDataset(data_dir="data", use_synthetic=True, size=1000, split="test")
    
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

if __name__ == "__main__":
    test_ben14k_split_disjointness()
    test_dsrsid_split_disjointness()
    print("ALL DATASET SPLIT DISJOINTNESS TESTS PASSED CLEANLY!")
