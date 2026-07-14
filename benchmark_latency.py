import time
import torch
import numpy as np
from Saber.utils.config import load_config
from Saber.models.saber import SABER
from Saber.retrieval.faiss_index import FAISSIndex

def benchmark_latency(config_path="Saber/configs/config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on Device: {device}")

    # 1. Load config
    config = load_config(config_path)

    # 2. Instantiate SABER
    model = SABER(config=config, in_channels=2).to(device)
    model.eval()

    # 3. Create dummy inputs (representing batch_size=1 query)
    dummy_query = torch.randn(1, 2, 224, 224).to(device)

    # 4. Create dummy gallery features for FAISS search (11,866 gallery items, 384 dim)
    num_gallery = 11866
    gallery_feats = np.random.randn(num_gallery, 384).astype(np.float32)
    faiss_index = FAISSIndex(dimension=384, metric="cosine")
    faiss_index.build_index(gallery_feats)

    # Warmup runs (to eliminate PyTorch lazy initialization overhead)
    print("Warming up model layers...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.get_retrieval_embedding(dummy_query)

    print("\nStarting Latency Benchmark (100 runs)...")
    backbone_times = []
    bridge_times = []
    faiss_times = []
    total_times = []

    for _ in range(100):
        t0 = time.perf_counter()
        
        with torch.no_grad():
            t_enc_start = time.perf_counter()
            feats = model.backbone(dummy_query, model.s1_wvs)
            z_s1 = model.projection_head(feats)
            t_enc_end = time.perf_counter()
            backbone_times.append((t_enc_end - t_enc_start) * 1000)

            # Time the Neural ODE bridge translation (S1 -> S2 space)
            t_bridge_start = time.perf_counter()
            if model.bridge is not None:
                query_embedding = model.bridge(z_s1)
            else:
                query_embedding = z_s1
            query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1)
            t_bridge_end = time.perf_counter()
            bridge_times.append((t_bridge_end - t_bridge_start) * 1000)

        # Convert to numpy array for FAISS
        query_np = query_embedding.cpu().numpy()

        # Time the FAISS index search lookup (retrieve top-5 nearest matches)
        t_faiss_start = time.perf_counter()
        _ = faiss_index.search(query_np, k=5)
        t_faiss_end = time.perf_counter()
        faiss_times.append((t_faiss_end - t_faiss_start) * 1000)

        t_total_end = time.perf_counter()
        total_times.append((t_total_end - t0) * 1000)

    # Calculate statistics
    mean_backbone = np.mean(backbone_times)
    mean_bridge = np.mean(bridge_times)
    mean_faiss = np.mean(faiss_times)
    mean_total = np.mean(total_times)

    print("=" * 45)
    print("          SABER LATENCY BENCHMARK          ")
    print("=" * 45)
    print(f"Backbone Encoder Pass: {mean_backbone:6.2f} ms")
    print(f"CFM Neural ODE Bridge: {mean_bridge:6.2f} ms")
    print(f"FAISS Database Lookup: {mean_faiss:6.2f} ms")
    print("-" * 45)
    print(f"TOTAL END-TO-END TIME: {mean_total:6.2f} ms")
    print("=" * 45)

if __name__ == "__main__":
    benchmark_latency()
