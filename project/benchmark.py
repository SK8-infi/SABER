import sys
import os
sys.path.append(os.getcwd())
import time
import torch
import numpy as np
import logging
import argparse
from torch.utils.data import DataLoader
from project.utils.config import load_config
from project.datasets.ben14k import BEN14KDataset
from project.datasets.dsrsid import DSRSIDDataset
from project.models.backbone import FrozenViTBackbone
from project.models.input_adapter import InputAdapter
from project.models.projection_head import ProjectionHead
from project.retrieval.faiss_index import FAISSIndex

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
    logger.info("Loading models...")
    backbone = FrozenViTBackbone(model_name=config.model.backbone_name, pretrained=False)
    backbone = backbone.to(device)
    feat_dim = backbone.embed_dim
    backbone.eval()
    
    adapter = InputAdapter(
        in_channels=dataset.num_channels,
        adapter_type=config.model.input_adapter_type
    ).to(device)
    adapter.eval()
    
    proj_head = ProjectionHead(
        in_dim=feat_dim,
        hidden_dim=config.model.projection_head.hidden_dim,
        out_dim=config.model.projection_head.out_dim
    ).to(device)
    proj_head.eval()

    logger.info("Warmup forward passes...")
    dummy_x = torch.randn(16, dataset.num_channels, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = proj_head(backbone(adapter(dummy_x)))

    logger.info("Measuring inference forward latency (50 iterations)...")
    inf_times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = proj_head(backbone(adapter(dummy_x)))
            if device.type == "cuda":
                torch.cuda.synchronize()
            inf_times.append((time.time() - t0) * 1000)
            
    avg_inf_time = np.mean(inf_times)
    inf_throughput = (16 * 1000) / avg_inf_time
    logger.info(f"Average Forward Inference Latency: {avg_inf_time:.2f} ms (batch=16) | Throughput: {inf_throughput:.2f} images/sec")

    # 3. Measure FAISS Index Retrieval Speed
    logger.info("Measuring FAISS Cosine retrieval latency...")
    # Load index if archived, otherwise build a dummy one of similar scale
    index_path = f"checkpoints/{dataset_name.lower()}/faiss_index.bin"
    if os.path.exists(index_path):
        faiss_idx = FAISSIndex(dimension=config.model.projection_head.out_dim)
        faiss_idx.load_index(index_path)
    else:
        # Fallback dummy index (12,000 items)
        faiss_idx = FAISSIndex(dimension=config.model.projection_head.out_dim)
        dummy_feats = np.random.randn(12000, config.model.projection_head.out_dim).astype(np.float32)
        # Normalize
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
        "gpu_memory_mb": max_memory
    }

def main():
    parser = argparse.ArgumentParser(description="REJEPA Pipeline Benchmarking Utility")
    parser.add_argument("--config", type=str, default="project/configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    results = []
    
    # Benchmark BEN-14K
    ben_dir = "C:/Users/praba/Downloads/benv1_14k"
    if os.path.exists(ben_dir):
        try:
            results.append(run_benchmark(args.config, "ben14k", ben_dir))
        except Exception as e:
            logger.error(f"Failed to benchmark BEN-14K: {e}")
            
    # Benchmark DSRSID
    dsrsid_file = "C:/Users/praba/Downloads/DSRSID.mat"
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
            ("Model Forward Latency (batch=16)", "inference_latency_ms", "{:.2f} ms"),
            ("Model Inference Throughput", "inference_throughput_ips", "{:.2f} img/s"),
            ("FAISS Top-5 Search Latency", "faiss_latency_ms", "{:.4f} ms"),
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
