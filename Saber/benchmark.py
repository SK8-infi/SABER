import sys
import os
sys.path.append(os.getcwd())
import time
import torch
import numpy as np
import logging
import argparse
from torch.utils.data import DataLoader
from Saber.utils.config import load_config
from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import FAISSIndex

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("benchmark")

def run_benchmark(config_path: str, dataset_name: str, data_dir: str):
    logger.info(f"=== BENCHMARKING SYSTEM FOR: {dataset_name.upper()} ===")
    config = load_config(config_path)
    config.dataset.name = dataset_name
    config.dataset.data_dir = data_dir
    config.dataset.use_synthetic = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Computation Device: {device}")
    
    # 1. Measure Data Loading Throughput
    logger.info("Initializing Dataset loader...")
    start_time = time.time()
    if dataset_name.lower() == "ben14k":
        dataset = BEN14KDataset(
            data_dir=data_dir,
            use_synthetic=False,
            size=1000,
            image_size=config.dataset.image_size,
            modality=config.dataset.get("modality", "s2"),
            is_train=False
        )
    else:
        dataset = DSRSIDDataset(
            data_dir=data_dir,
            use_synthetic=False,
            size=1000,
            image_size=config.dataset.image_size,
            is_train=False
        )
    load_init_time = (time.time() - start_time) * 1000
    logger.info(f"Dataset initialized in {load_init_time:.2f} ms")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Measure IO disk read speed
    io_times = []
    logger.info("Measuring disk data ingestion latency (10 batches)...")
    iterator = iter(loader)
    for _ in range(10):
        t0 = time.time()
        try:
            batch = next(iterator)
            io_times.append((time.time() - t0) * 1000)
        except StopIteration:
            break
            
    avg_io_time = np.mean(io_times)
    throughput_ips = (32 * 1000) / avg_io_time
    logger.info(f"Average IO Batch Load Time: {avg_io_time:.2f} ms | Throughput: {throughput_ips:.2f} images/sec")

    # 2. Measure Model Forward Inference Latency
    logger.info("Loading SABER model...")
    # Force bridge enabled for latency profiling
    config.bridge.enabled = True
    if dataset_name.lower() == "dsrsid":
        config.bridge.checkpoint = "checkpoints/bridge_best_dsrsid.pth"
        checkpoint_path = "checkpoints/latest_dsrsid.pth"
    else:
        config.bridge.checkpoint = "checkpoints/bridge_best.pth"
        checkpoint_path = "checkpoints/latest_ben14k.pth"

    model_in_channels = 14 if dataset_name.lower() == "ben14k" else 5
    model = SABER(config=config, in_channels=model_in_channels).to(device)
    model.eval()

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded model checkpoint: {checkpoint_path}")
    if os.path.exists(config.bridge.checkpoint):
        model.bridge.cfm_bridge.load_state_dict(torch.load(config.bridge.checkpoint, map_location=device, weights_only=True))
        logger.info(f"Loaded CFM Latent Bridge: {config.bridge.checkpoint}")

    # Determine S1 query channels
    s1_channels = model.s1_channels

    logger.info("Warmup forward passes...")
    dummy_x = torch.randn(1, s1_channels, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model.get_retrieval_embedding(dummy_x)

    logger.info("Measuring model forward latency (50 iterations)...")
    inf_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = model.get_retrieval_embedding(dummy_x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_times.append((time.time() - t0) * 1000)
            
    avg_inf_time = np.mean(inf_times)
    inf_throughput = 1000.0 / avg_inf_time
    logger.info(f"Average Single-Query Forward Latency (with CFM Bridge): {avg_inf_time:.2f} ms | Throughput: {inf_throughput:.2f} queries/sec")

    # 3. Measure FAISS Index Retrieval Speed
    logger.info("Measuring FAISS Cosine retrieval latency...")
    # Load index if archived, otherwise build a dummy one of similar scale
    index_path = f"checkpoints/{dataset_name.lower()}/faiss_index.bin"
    if os.path.exists(index_path):
        faiss_idx = FAISSIndex(dimension=config.model.projection_head.out_dim)
        faiss_idx.load_index(index_path)
    else:
        # Fallback dummy index (10,000 items)
        faiss_idx = FAISSIndex(dimension=config.model.projection_head.out_dim)
        dummy_feats = np.random.randn(10000, config.model.projection_head.out_dim).astype(np.float32)
        dummy_feats = dummy_feats / np.linalg.norm(dummy_feats, axis=1, keepdims=True)
        faiss_idx.build_index(dummy_feats)
        
    query_feats = np.random.randn(100, config.model.projection_head.out_dim).astype(np.float32)
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1, keepdims=True)
    
    retrieval_times = []
    for q in query_feats:
        t0 = time.time()
        _, _ = faiss_idx.search(q.reshape(1, -1), k=5)
        retrieval_times.append((time.time() - t0) * 1000)
        
    avg_ret_time = np.mean(retrieval_times)
    logger.info(f"Average FAISS Top-5 Cosine Search Latency: {avg_ret_time:.4f} ms per query")
    
    # Calculate end-to-end average retrieval time per query
    avg_e2e_time = avg_inf_time + avg_ret_time
    logger.info(f"Average End-to-End Retrieval Latency (Model + Search): {avg_e2e_time:.2f} ms")

    # 4. Memory Profiling
    max_memory = 0
    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logger.info(f"Peak GPU VRAM Allocated: {max_memory:.2f} MB")
        
    return {
        "dataset": dataset_name.upper(),
        "io_latency_ms": avg_io_time,
        "io_throughput_ips": throughput_ips,
        "inference_latency_ms": avg_inf_time,
        "inference_throughput_ips": inf_throughput,
        "faiss_latency_ms": avg_ret_time,
        "e2e_latency_ms": avg_e2e_time,
        "gpu_memory_mb": max_memory
    }

def main():
    parser = argparse.ArgumentParser(description="REJEPA Pipeline Benchmarking Utility")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    results = []
    
    # Benchmark BEN-14K
    ben_dir = "c:/Github/SABER/Datasets/benv1_14k"
    if os.path.exists(ben_dir):
        try:
            results.append(run_benchmark(args.config, "ben14k", ben_dir))
        except Exception as e:
            logger.error(f"Failed to benchmark BEN-14K: {e}")
            
    # Benchmark DSRSID
    dsrsid_file = "c:/Github/SABER/Datasets/DSRSID/DSRSID-001.mat"
    if os.path.exists(dsrsid_file):
        try:
            results.append(run_benchmark(args.config, "dsrsid", dsrsid_file))
        except Exception as e:
            logger.error(f"Failed to benchmark DSRSID: {e}")

    # Output Benchmark Table
    if results:
        print("\n" + "="*80)
        print("                        BENCHMARK PERFORMANCE REPORT")
        print("="*80)
        print(f"{'Metric':<40} | {'BEN-14K':<15} | {'DSRSID (Gaofen-1)':<15}")
        print("-"*80)
        
        # Helper to get value
        def get_val(dataset_name, key):
            for r in results:
                if r["dataset"].lower() == dataset_name.lower():
                    return r[key]
            return None

        # Print metrics rows
        keys = [
            ("Average Batch Ingestion IO (32 images)", "io_latency_ms", "{:.2f} ms"),
            ("Data Ingestion Throughput", "io_throughput_ips", "{:.2f} img/s"),
            ("Single-Query Forward Latency (w/ CFM Bridge)", "inference_latency_ms", "{:.2f} ms"),
            ("Query Extraction Throughput", "inference_throughput_ips", "{:.2f} queries/s"),
            ("FAISS Top-5 Search Latency", "faiss_latency_ms", "{:.4f} ms"),
            ("Average End-to-End Latency per Query", "e2e_latency_ms", "{:.2f} ms"),
            ("Peak GPU VRAM Allocation", "gpu_memory_mb", "{:.2f} MB")
        ]
        
        for name, key, fmt in keys:
            v_ben = get_val("ben14k", key)
            v_ds = get_val("dsrsid", key)
            
            s_ben = fmt.format(v_ben) if v_ben is not None else "N/A"
            s_ds = fmt.format(v_ds) if v_ds is not None else "N/A"
            print(f"{name:<40} | {s_ben:<15} | {s_ds:<15}")
        print("="*80)

if __name__ == "__main__":
    main()
