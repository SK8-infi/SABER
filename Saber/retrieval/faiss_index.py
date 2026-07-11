import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger("saber")


@dataclass
class SearchProfile:
    avg_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    num_queries: int


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def pack_binary_codes(codes: np.ndarray) -> np.ndarray:
    """
    Converts {-1,+1}, {0,1}, or real-valued hash codes to FAISS uint8 packs.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError("Binary codes must have shape (N, num_bits).")
    if codes.shape[1] % 8 != 0:
        raise ValueError("Binary code dimension must be divisible by 8.")

    bits = codes > 0
    return np.ascontiguousarray(np.packbits(bits.astype(np.uint8), axis=1))


def random_projection_codes(embeddings: np.ndarray, num_bits: int = 256, seed: int = 42) -> np.ndarray:
    """
    Zero-training fallback for binary indexing. A learned HashingHead should
    replace this when training integration is ready.
    """
    embeddings = l2_normalize(embeddings)
    rng = np.random.RandomState(seed)
    projection = rng.normal(size=(embeddings.shape[1], num_bits)).astype(np.float32)
    projection /= np.maximum(np.linalg.norm(projection, axis=0, keepdims=True), 1e-12)
    return embeddings @ projection


class AdvancedFAISSIndex:
    """
    FAISS wrapper for Dev 4 retrieval experiments.

    Supported index types:
    - flat: exact float search, baseline correctness path
    - ivfpq: compressed approximate float search
    - binary_hnsw: Hamming search over packed binary codes
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "flat",
        nlist: int = 64,
        pq_m: int = 64,
        pq_bits: int = 4,
        hnsw_m: int = 32,
        nprobe: int = 8,
        hash_bits: int = 256,
        fast_scan: bool = False,
        seed: int = 42,
    ) -> None:
        self.dimension = dimension
        self.metric = metric.lower()
        self.index_type = index_type.lower()
        self.nlist = nlist
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.hnsw_m = hnsw_m
        self.nprobe = nprobe
        self.hash_bits = hash_bits
        self.fast_scan = fast_scan
        self.seed = seed
        self.index = None
        self._projection = None

        if self.metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'.")
        if self.index_type not in {"flat", "ivfpq", "binary_hnsw"}:
            raise ValueError("index_type must be 'flat', 'ivfpq', or 'binary_hnsw'.")

    def _prepare_float(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {embeddings.shape[1]}.")
        if self.metric == "cosine":
            embeddings = l2_normalize(embeddings)
        return embeddings

    def _make_binary_codes(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = self._prepare_float(embeddings)
        if self._projection is None:
            rng = np.random.RandomState(self.seed)
            self._projection = rng.normal(size=(self.dimension, self.hash_bits)).astype(np.float32)
            self._projection /= np.maximum(np.linalg.norm(self._projection, axis=0, keepdims=True), 1e-12)
        return pack_binary_codes(embeddings @ self._projection)

    def build_index(self, embeddings: np.ndarray, binary_codes: Optional[np.ndarray] = None) -> None:
        if self.index_type == "binary_hnsw":
            packed = pack_binary_codes(binary_codes) if binary_codes is not None else self._make_binary_codes(embeddings)
            self.index = faiss.IndexBinaryHNSW(self.hash_bits, self.hnsw_m)
            self.index.add(packed)
            logger.info("Built FAISS IndexBinaryHNSW with %d items and %d bits.", self.index.ntotal, self.hash_bits)
            return

        vectors = self._prepare_float(embeddings)
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors)
            logger.info("Built FAISS flat %s index with %d items.", self.metric, self.index.ntotal)
            return

        train_count = vectors.shape[0]
        nlist = max(1, min(self.nlist, train_count))
        pq_m = min(self.pq_m, self.dimension)
        while self.dimension % pq_m != 0 and pq_m > 1:
            pq_m -= 1

        if train_count < max(32, nlist * 2):
            logger.warning("Only %d vectors available; falling back from IVFPQ to flat index.", train_count)
            self.index_type = "flat"
            self.index = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
            self.index.add(vectors)
            return

        faiss_metric = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
        quantizer = faiss.IndexFlatIP(self.dimension) if self.metric == "cosine" else faiss.IndexFlatL2(self.dimension)
        if self.fast_scan and self.pq_bits == 4 and hasattr(faiss, "IndexIVFPQFastScan"):
            self.index = faiss.IndexIVFPQFastScan(quantizer, self.dimension, nlist, pq_m, self.pq_bits, faiss_metric)
        else:
            if self.fast_scan and self.pq_bits != 4:
                logger.warning("FastScan requires 4-bit PQ; falling back to standard IndexIVFPQ.")
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, pq_m, self.pq_bits, faiss_metric)
        self.index.train(vectors)
        self.index.add(vectors)
        self.index.nprobe = min(self.nprobe, nlist)
        logger.info(
            "Built FAISS IndexIVFPQ with %d items, nlist=%d, pq_m=%d, bits=%d, nprobe=%d.",
            self.index.ntotal,
            nlist,
            pq_m,
            self.pq_bits,
            self.index.nprobe,
        )

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("FAISS index has not been built or loaded.")
        if self.index_type == "binary_hnsw":
            queries = self._make_binary_codes(query_embeddings)
        else:
            queries = self._prepare_float(query_embeddings)
        return self.index.search(queries, k)

    def profile_search(self, query_embeddings: np.ndarray, k: int = 5, repeats: int = 1) -> SearchProfile:
        timings = []
        for _ in range(repeats):
            for query in query_embeddings:
                start = time.perf_counter()
                self.search(query.reshape(1, -1), k=k)
                timings.append((time.perf_counter() - start) * 1000.0)
        values = np.asarray(timings, dtype=np.float64)
        return SearchProfile(
            avg_ms=float(np.mean(values)),
            p95_ms=float(np.percentile(values, 95)),
            min_ms=float(np.min(values)),
            max_ms=float(np.max(values)),
            num_queries=int(len(values)),
        )

    def save_index(self, path: str) -> None:
        if self.index is None:
            raise ValueError("No active FAISS index to save.")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self.index_type == "binary_hnsw":
            faiss.write_index_binary(self.index, path)
        else:
            faiss.write_index(self.index, path)

    def load_index(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if self.index_type == "binary_hnsw":
            self.index = faiss.read_index_binary(path)
        else:
            self.index = faiss.read_index(path)


# Backward compatibility alias
FAISSIndex = AdvancedFAISSIndex
