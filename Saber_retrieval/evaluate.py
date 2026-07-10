import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from Saber.datasets.ben14k import BEN14KDataset
from Saber.datasets.dsrsid import DSRSIDDataset
from Saber.datasets.transforms import get_transforms
from Saber.models.rejepa import REJEPA
from Saber.trainer.evaluator import Evaluator
from Saber.utils.checkpoint import load_checkpoint
from Saber.utils.config import load_config
from Saber.utils.logger import setup_logger
from Saber.utils.seed import set_seed
from Saber_retrieval.retrieval.faiss_index import AdvancedFAISSIndex
from Saber_retrieval.retrieval.rerank import ReciprocalReranker
from Saber_retrieval.utils.metrics import metrics_from_ranked_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Developer 4 compact retrieval evaluation")
    parser.add_argument("--config", type=str, default="Saber/configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--synthetic", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--modality", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "ivfpq", "binary_hnsw"])
    parser.add_argument("--metric", type=str, default=None, choices=["cosine", "l2"])
    parser.add_argument("--nlist", type=int, default=64)
    parser.add_argument("--pq_m", type=int, default=64)
    parser.add_argument("--pq_bits", type=int, default=4)
    parser.add_argument("--nprobe", type=int, default=8)
    parser.add_argument("--hash_bits", type=int, default=256)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--fast_scan", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--shortlist_k", type=int, default=100)
    parser.add_argument("--rerank_timeout_ms", type=float, default=50.0)
    parser.add_argument("--latency_queries", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="Saber_retrieval/outputs")
    return parser.parse_args()


def apply_overrides(config: Any, args: argparse.Namespace) -> Any:
    if args.synthetic is not None:
        config.dataset.use_synthetic = args.synthetic.lower() == "true"
    if args.dataset_name is not None:
        config.dataset.name = args.dataset_name
    if args.data_dir is not None:
        config.dataset.data_dir = args.data_dir
    if args.modality is not None:
        config.dataset.modality = args.modality
    if args.metric is not None:
        config.retrieval.metric = args.metric
    return config


def build_dataset(config: Any) -> Tuple[Any, int]:
    transform = get_transforms(image_size=config.dataset.image_size, is_train=False)
    dataset_name = config.dataset.name.lower()
    if dataset_name == "ben14k":
        dataset = BEN14KDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=transform,
            modality=config.dataset.get("modality", "s2"),
            is_train=False,
        )
        return dataset, dataset.num_channels
    if dataset_name == "dsrsid":
        dataset = DSRSIDDataset(
            data_dir=config.dataset.data_dir,
            use_synthetic=config.dataset.use_synthetic,
            size=config.dataset.get("size", 1000),
            image_size=config.dataset.image_size,
            transform=transform,
            is_train=False,
        )
        return dataset, dataset.num_channels
    raise ValueError(f"Unknown dataset '{config.dataset.name}'.")


def compute_rankings(
    index: AdvancedFAISSIndex,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int,
    shortlist_k: int,
    use_rerank: bool,
    rerank_timeout_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    search_k = max(top_k, shortlist_k if use_rerank else top_k)
    scores, indices = index.search(query_embeddings, k=search_k)
    if not use_rerank:
        return scores[:, :top_k], indices[:, :top_k]

    reranker = ReciprocalReranker(shortlist_k=shortlist_k)
    reranked_scores = []
    reranked_indices = []
    start = time.perf_counter()
    for q_idx in range(query_embeddings.shape[0]):
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > rerank_timeout_ms:
            return scores[:, :top_k], indices[:, :top_k]
        q_scores, q_indices = reranker.rerank(
            query_embedding=query_embeddings[q_idx],
            gallery_embeddings=gallery_embeddings,
            indices=indices[q_idx],
            scores=scores[q_idx],
            gallery_labels=gallery_labels,
            uncertainty=0.0,
            final_k=top_k,
        )
        reranked_scores.append(q_scores)
        reranked_indices.append(q_indices)
    return np.asarray(reranked_scores), np.asarray(reranked_indices)


def profile_end_to_end_retrieval(
    index: AdvancedFAISSIndex,
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    top_k: int,
    shortlist_k: int,
    use_rerank: bool,
    rerank_timeout_ms: float,
) -> Dict[str, float]:
    timings = []
    search_k = max(top_k, shortlist_k if use_rerank else top_k)
    reranker = ReciprocalReranker(shortlist_k=shortlist_k) if use_rerank else None

    for query in query_embeddings:
        start = time.perf_counter()
        scores, indices = index.search(query.reshape(1, -1), k=search_k)
        if reranker is not None and ((time.perf_counter() - start) * 1000.0) <= rerank_timeout_ms:
            reranker.rerank(
                query_embedding=query,
                gallery_embeddings=gallery_embeddings,
                indices=indices[0],
                scores=scores[0],
                gallery_labels=gallery_labels,
                uncertainty=0.0,
                final_k=top_k,
            )
        timings.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(timings, dtype=np.float64)
    return {
        "avg_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "num_queries": int(len(values)),
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    logger = setup_logger(name="saber", log_dir=config.log_dir)
    set_seed(config.seed)

    device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    logger.info("Developer 4 retrieval evaluation starting on device: %s", device)

    dataset, in_channels = build_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=0 if config.dataset.name.lower() == "dsrsid" else config.dataset.num_workers,
    )

    model = REJEPA(config=config, in_channels=in_channels).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = load_checkpoint(args.checkpoint, map_location=str(device))
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.warning("No checkpoint found/provided; evaluating retrieval with initialized weights.")

    evaluator = Evaluator(model=model, dataloader=loader, device=device, config=config)
    dense_results = evaluator.evaluate(top_k=args.top_k)

    query_indices = dense_results["query_indices"]
    gallery_indices = dense_results["gallery_indices"]
    embeddings = dense_results["embeddings"].astype(np.float32)
    labels = dense_results["labels"]
    names = dense_results["names"]

    query_embeddings = embeddings[query_indices]
    gallery_embeddings = embeddings[gallery_indices]
    query_labels = labels[query_indices]
    gallery_labels = labels[gallery_indices]
    gallery_names = [names[i] for i in gallery_indices]

    index = AdvancedFAISSIndex(
        dimension=gallery_embeddings.shape[1],
        metric=config.retrieval.metric,
        index_type=args.index_type,
        nlist=args.nlist,
        pq_m=args.pq_m,
        pq_bits=args.pq_bits,
        hnsw_m=args.hnsw_m,
        nprobe=args.nprobe,
        hash_bits=args.hash_bits,
        fast_scan=args.fast_scan,
        seed=config.seed,
    )

    build_start = time.perf_counter()
    index.build_index(gallery_embeddings)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    scores, ranked_indices = compute_rankings(
        index=index,
        query_embeddings=query_embeddings,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        top_k=args.top_k,
        shortlist_k=args.shortlist_k,
        use_rerank=args.rerank,
        rerank_timeout_ms=args.rerank_timeout_ms,
    )
    is_multilabel = config.dataset.name.lower() == "ben14k"
    faiss_metrics = metrics_from_ranked_indices(
        ranked_indices=ranked_indices,
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        top_k=args.top_k,
        is_multilabel=is_multilabel,
    )

    latency_sample = query_embeddings[: max(1, min(args.latency_queries, query_embeddings.shape[0]))]
    profile = index.profile_search(latency_sample, k=max(args.top_k, args.shortlist_k if args.rerank else args.top_k))
    end_to_end_profile = profile_end_to_end_retrieval(
        index=index,
        query_embeddings=latency_sample,
        gallery_embeddings=gallery_embeddings,
        gallery_labels=gallery_labels,
        top_k=args.top_k,
        shortlist_k=args.shortlist_k,
        use_rerank=args.rerank,
        rerank_timeout_ms=args.rerank_timeout_ms,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    index_path = os.path.join(args.output_dir, f"{args.index_type}.faiss")
    index.save_index(index_path)
    index_size_mb = os.path.getsize(index_path) / (1024 * 1024)

    metadata_path = os.path.join(args.output_dir, f"{args.index_type}_metadata.pth")
    torch.save(
        {
            "gallery_names": gallery_names,
            "gallery_labels": gallery_labels,
            "gallery_indices": gallery_indices,
            "index_type": args.index_type,
            "metric": config.retrieval.metric,
            "hash_bits": args.hash_bits,
            "binary_projection": getattr(index, "_projection", None),
        },
        metadata_path,
    )

    report: Dict[str, Any] = {
        "index_type": args.index_type,
        "metric": config.retrieval.metric,
        "rerank": args.rerank,
        "fast_scan": args.fast_scan,
        "top_k": args.top_k,
        "num_queries": int(query_embeddings.shape[0]),
        "num_gallery": int(gallery_embeddings.shape[0]),
        "build_ms": build_ms,
        "index_size_mb": index_size_mb,
        "faiss_latency": profile.__dict__,
        "end_to_end_latency": end_to_end_profile,
        "dense_baseline_metrics": dense_results["metrics"],
        "faiss_metrics": faiss_metrics,
        "index_path": index_path,
        "metadata_path": metadata_path,
    }
    report_path = os.path.join(args.output_dir, f"{args.index_type}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("========== DEV 4 RETRIEVAL REPORT ==========")
    logger.info("Index: %s | metric=%s | rerank=%s", args.index_type, config.retrieval.metric, args.rerank)
    logger.info("Build time: %.2f ms | index size: %.3f MB", build_ms, index_size_mb)
    logger.info(
        "FAISS latency: avg %.4f ms | p95 %.4f ms | min %.4f ms | max %.4f ms over %d queries",
        profile.avg_ms,
        profile.p95_ms,
        profile.min_ms,
        profile.max_ms,
        profile.num_queries,
    )
    logger.info(
        "End-to-end retrieval latency: avg %.4f ms | p95 %.4f ms | min %.4f ms | max %.4f ms over %d queries",
        end_to_end_profile["avg_ms"],
        end_to_end_profile["p95_ms"],
        end_to_end_profile["min_ms"],
        end_to_end_profile["max_ms"],
        end_to_end_profile["num_queries"],
    )
    for name, value in faiss_metrics.items():
        logger.info("FAISS %s: %.4f", name, value)
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    main()
