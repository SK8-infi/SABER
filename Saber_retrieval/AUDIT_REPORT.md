# Developer 4 Audit Report

## Scope

This audit reviews the `Saber_retrieval/` implementation against:

- `docs/plan.md`
- `docs/risk.md`
- `docs/split.md`
- `docs/implementation_plan.md`
- `docs/saber_benchmarking_report.md`
- Architecture PDFs present in the repo

`pdftotext`, `PyPDF2`, `pypdf`, and `pdfplumber` were not available in the local environment. PDF validation was therefore limited to filename review and coarse `strings` scanning. The slide deck exposed titles for "Proposed System Architecture" and "RETRIEVAL LAYER", consistent with the Markdown architecture docs.

## Architecture Compliance

| Requirement | Status | Evidence |
|---|---:|---|
| Hashing head with tanh relaxation | Complete | `models/hashing_head.py` implements soft tanh codes, hard sign codes, and quantization loss. |
| Similarity-preserving hash objective | Complete | `similarity_preserving_hash_loss` supports multi-label and single-label supervision. |
| FAISS flat baseline | Complete | `AdvancedFAISSIndex(index_type="flat")`. |
| FAISS `IndexIVFPQ` | Complete | `AdvancedFAISSIndex(index_type="ivfpq")`. |
| FAISS IVFPQ FastScan | Complete | `--fast_scan` uses `faiss.IndexIVFPQFastScan` when available and PQ bits are 4. |
| FAISS `IndexBinaryHNSW` | Complete | `AdvancedFAISSIndex(index_type="binary_hnsw")`. |
| Graph/k-reciprocal shortlist re-ranking | Complete as optional path | `retrieval/rerank.py`; enabled with `--rerank`. |
| Risk fallback for slow re-ranking | Complete | `--rerank_timeout_ms` falls back to raw FAISS rankings. |
| Latency profiler | Complete | Reports raw FAISS latency and end-to-end retrieval latency. |
| Index size/build-time reporting | Complete | `evaluate.py` writes JSON reports with build time and index size. |

## Senior SDE Findings

### Fixed During Audit

1. **Re-ranker one-item shortlist edge case**
   - Issue: A single candidate caused an empty neighbor mean and potential NaN.
   - Fix: Early return for one-candidate shortlists.

2. **Missing re-ranking timeout fallback**
   - Issue: Initial implementation made re-ranking optional but did not enforce the timeout recommended in `risk.md`.
   - Fix: Added `--rerank_timeout_ms`; fallback returns raw FAISS rankings.

3. **IVFPQ FastScan gap**
   - Issue: Initial implementation used standard `IndexIVFPQ`; `plan.md` asks for FastScan.
   - Fix: Added `--fast_scan` and runtime FAISS capability check.

4. **Latency report ambiguity**
   - Issue: Initial latency metric measured raw FAISS only, even when re-ranking was enabled.
   - Fix: Report now includes both `faiss_latency` and `end_to_end_latency`.

### Residual Risks

1. **Binary HNSW currently uses random projection fallback unless a trained `HashingHead` checkpoint is integrated.**
   - This is acceptable for zero-shot infrastructure testing, but final accuracy requires training the hashing head with model embeddings.

2. **mAP from FAISS-ranked shortlists is shortlist mAP, not full-gallery mAP.**
   - Dense baseline metrics still come from the existing evaluator. Dev 4 FAISS metrics are meant to compare retrieval index behavior.

3. **OpenMP conflict on macOS**
   - End-to-end runs crashed without `KMP_DUPLICATE_LIB_OK=TRUE` due duplicate `libomp.dylib` initialization between ML/FAISS dependencies.
   - This is an environment/runtime packaging issue, not a logic error.

4. **Tiny synthetic IVFPQ tests produce FAISS clustering warnings.**
   - Expected because IVFPQ needs more training vectors. Production/gallery runs should use larger datasets.

## Test Results

### Static/Compilation

Command:

```bash
Saber/.venv/bin/python -m py_compile Saber_retrieval/evaluate.py Saber_retrieval/models/hashing_head.py Saber_retrieval/retrieval/faiss_index.py Saber_retrieval/retrieval/rerank.py Saber_retrieval/utils/metrics.py
```

Result: Passed.

### Component Tests

Random embeddings were used to test:

- flat FAISS search
- IVFPQ FastScan search
- binary HNSW search
- single-candidate re-ranker edge case
- metric calculation sanity

Result: Passed.

### End-to-End Synthetic Tests

Tests used a temporary config:

- `pretrained: false`
- `dataset.size: 40`
- `batch_size: 8`
- synthetic data enabled
- CPU execution

Because of macOS OpenMP runtime duplication, commands were run with:

```bash
env KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
```

#### Flat FAISS

- Build time: `0.29 ms`
- Index size: `0.047 MB`
- FAISS latency avg: `0.0128 ms`
- FAISS p95: `0.0232 ms`
- Precision@5: `0.1292`
- F1@5: `0.1000`

#### Binary HNSW

- Build time: `28.61 ms`
- Index size: `0.009 MB`
- FAISS latency avg: `0.1618 ms`
- FAISS p95: `0.5248 ms`
- Precision@5: `0.1042`
- F1@5: `0.0967`

#### IVFPQ FastScan

- Build time: `9.38 ms`
- Index size: `0.027 MB`
- FAISS latency avg: `0.0299 ms`
- FAISS p95: `0.0554 ms`
- Precision@5: `0.0958`
- F1@5: `0.0783`

#### Optional Re-ranking

- Build time: `0.53 ms`
- Index size: `0.047 MB`
- FAISS latency avg: `0.0105 ms`
- End-to-end retrieval latency avg: `0.0771 ms`
- End-to-end p95: `0.0855 ms`
- Precision@5: `0.0917`
- F1@5: `0.0775`

#### Timeout Fallback

`--rerank_timeout_ms 0` returned raw FAISS-equivalent metrics, confirming fallback behavior.

## Verdict

The Developer 4 retrieval module is architecturally aligned with the project docs and has passed focused component tests plus small end-to-end synthetic evaluation. The implementation is ready for teammate review and larger-gallery benchmarking.

The next engineering step is to connect the learned `HashingHead` to training/checkpointing so binary HNSW uses trained semantic hash codes instead of random projection fallback.

