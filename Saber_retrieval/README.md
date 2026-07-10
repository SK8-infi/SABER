# Saber Retrieval - Developer 4

This workspace implements the compact retrieval lane from `docs/plan.md`:
binary hashing, FAISS indexing, latency profiling, and optional graph re-ranking.

## Implemented

- `models/hashing_head.py`: train-time tanh relaxed hash codes and hash loss.
- `retrieval/faiss_index.py`: `flat`, `IndexIVFPQ`, and `IndexBinaryHNSW` wrappers.
- `retrieval/rerank.py`: bounded reciprocal-neighbor shortlist re-ranker.
- `evaluate.py`: end-to-end retrieval evaluation using baseline SABER embeddings.

## Recommended MVP Commands

Run from the repository root:

```bash
Saber/.venv/bin/python Saber_retrieval/evaluate.py --checkpoint checkpoints/latest.pth --synthetic true --index_type flat
```

Test compressed float retrieval:

```bash
Saber/.venv/bin/python Saber_retrieval/evaluate.py --checkpoint checkpoints/latest.pth --synthetic true --index_type ivfpq --nlist 16 --pq_m 32 --fast_scan
```

Test binary HNSW retrieval:

```bash
Saber/.venv/bin/python Saber_retrieval/evaluate.py --checkpoint checkpoints/latest.pth --synthetic true --index_type binary_hnsw --hash_bits 256
```

Test optional re-ranking over a shortlist:

```bash
Saber/.venv/bin/python Saber_retrieval/evaluate.py --checkpoint checkpoints/latest.pth --synthetic true --index_type flat --rerank --shortlist_k 100
```

## Notes

`risk.md` recommends making re-ranking optional because it can violate the
latency budget. Keep raw FAISS as the default fast path and use re-ranking only
when its measured F1 gain justifies the added latency.
