import time
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Gallery Embeddings onto GPU FP16 / FP32
meta = torch.load("checkpoints/crossmodal/faiss_index_metadata.pth", map_location="cpu", weights_only=False)
gallery_gpu = torch.from_numpy(meta["embeddings"]).to(device).float()
gallery_gpu_fp16 = gallery_gpu.half()

query_vec = torch.randn(1, 768, device=device).float()
query_vec_fp16 = query_vec.half()

# Warmup
for _ in range(10):
    _ = torch.mm(query_vec, gallery_gpu.T)
    _ = torch.mm(query_vec_fp16, gallery_gpu_fp16.T)
torch.cuda.synchronize()

# Test 1: CPU NumPy dot
emb_np = query_vec.cpu().numpy()
gal_np = meta["embeddings"]
t0 = time.perf_counter_ns()
for _ in range(100):
    _ = np.dot(emb_np, gal_np.T)
t1 = time.perf_counter_ns()
lat_cpu = (t1 - t0) / 100 / 1e6

# Test 2: GPU PyTorch Float32 Matmul
t0 = time.perf_counter_ns()
for _ in range(100):
    _ = torch.mm(query_vec, gallery_gpu.T)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()
lat_gpu_fp32 = (t1 - t0) / 100 / 1e6

# Test 3: GPU PyTorch FP16 Matmul
t0 = time.perf_counter_ns()
for _ in range(100):
    _ = torch.mm(query_vec_fp16, gallery_gpu_fp16.T)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()
lat_gpu_fp16 = (t1 - t0) / 100 / 1e6

print("=== LATENCY BENCHMARK: FAISS/SEARCH OPTIMIZATION ===")
print(f"NumPy CPU Matrix Search:     {lat_cpu:.3f} ms")
print(f"PyTorch CUDA FP32 Matmul:    {lat_gpu_fp32:.3f} ms  (Speedup: {lat_cpu/lat_gpu_fp32:.1f}x)")
print(f"PyTorch CUDA FP16 Matmul:    {lat_gpu_fp16:.3f} ms  (Speedup: {lat_cpu/lat_gpu_fp16:.1f}x)")
