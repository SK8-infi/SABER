import os
import sys
import time
import argparse
import torch
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_local_search(
    db_path: str = "saber_search_db.pth",
    query_name: str = None,
    query_class: str = None,
    mode: str = "cross",     # 'cross' (S1 -> S2), 'same' (S2 -> S2), 'hybrid' (0.4z+0.6p)
    top_k: int = 5
):
    print("=" * 80)
    print(" ⚡ SABER LOCAL EMBEDDING SEARCH ENGINE (ZERO GPU / ZERO PYTORCH NEEDED)")
    print("=" * 80)

    if not os.path.exists(db_path):
        print(f"❌ Database file '{db_path}' not found!")
        print("Please run 'python Saber/export_embeddings.py' on Colab first and download 'saber_search_db.pth'.")
        return

    # Load DB
    start_load = time.time()
    db = torch.load(db_path, map_location="cpu", weights_only=False)
    load_elapsed = time.time() - start_load

    N = db["num_samples"]
    names = list(db["names"])
    labels = db["labels"]
    class_names = db.get("class_names", [])

    print(f"✅ Loaded Database '{db_path}' in {load_elapsed:.3f}s | {N} Total Sample Vectors")

    # Name-to-Index Lookup Dictionary (Instant O(1) lookup!)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    # Determine Search Tensors
    s1_embeds = db["s1_translated_embeds"].astype(np.float32) if "s1_translated_embeds" in db else db["s1_embeds"].astype(np.float32)
    s2_embeds = db["s2_embeds"].astype(np.float32)
    h1_embeds = db["hybrid_s1_embeds"].astype(np.float32) if "hybrid_s1_embeds" in db else s1_embeds
    h2_embeds = db["hybrid_s2_embeds"].astype(np.float32) if "hybrid_s2_embeds" in db else s2_embeds

    # Mode selection
    if mode == "cross":
        query_pool = s1_embeds
        gallery_pool = s2_embeds
        mode_desc = "Cross-Modal SAR (S1) -> Optical (S2) Search [CFM Bridge ODE Translated]"
    elif mode == "hybrid":
        query_pool = h1_embeds
        gallery_pool = h2_embeds
        mode_desc = "ISRO Hybrid Search (0.4 Visual Embedding + 0.6 Land-Cover Semantics)"
    else:
        query_pool = s2_embeds
        gallery_pool = s2_embeds
        mode_desc = "Same-Modal Optical (S2) -> Optical (S2) Search"

    print(f"🔍 Search Mode: {mode_desc}")
    print("-" * 80)

    # 1. Search by Query Image Name
    if query_name is not None:
        if query_name not in name_to_idx:
            # Try fuzzy match
            matches = [n for n in names if query_name.lower() in n.lower()]
            if matches:
                query_name = matches[0]
                print(f"💡 Query matched to dataset sample: '{query_name}'")
            else:
                print(f"❌ Sample '{query_name}' not found in database!")
                print(f"Sample names example: {names[:5]}")
                return

        q_idx = name_to_idx[query_name]
        q_vec = query_pool[q_idx : q_idx + 1]  # (1, 768)

        # Microsecond Cosine Similarity Search
        start_search = time.time()
        scores = (query_pool[q_idx : q_idx + 1] @ gallery_pool.T)[0]  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k + 1]

        # Exclude self match
        results = [idx for idx in top_indices if idx != q_idx][:top_k]
        search_elapsed = (time.time() - start_search) * 1000

        print(f"🎯 Query Sample: '{query_name}' (Index #{q_idx})")
        print(f"⚡ Search Completed in {search_elapsed:.2f} ms")
        print("=" * 80)
        print(f"{'Rank':<6} | {'Similarity':<12} | {'Matched Image Name':<45}")
        print("-" * 80)

        for rank, idx in enumerate(results, 1):
            sim = scores[idx]
            match_name = names[idx]
            print(f"#{rank:<5} | {sim:<12.4f} | {match_name:<45}")

        print("=" * 80 + "\n")

    # 2. Search by Land-Cover Class Tag
    elif query_class is not None:
        class_idx = -1
        for i, cname in enumerate(class_names):
            if query_class.lower() in cname.lower():
                class_idx = i
                query_class = cname
                break

        if class_idx == -1:
            print(f"❌ Class tag '{query_class}' not found! Available classes:\n{class_names}")
            return

        print(f"🏷️ Searching all images containing Land-Cover Class: '{query_class}'...")
        matching_indices = np.where(labels[:, class_idx] > 0.5)[0]

        print(f"✅ Found {len(matching_indices)} matching images out of {N} total!")
        print("Top 10 Sample Matches:")
        for idx in matching_indices[:top_k]:
            print(f"  - {names[idx]}")

    else:
        # Default fallback: Pick random query
        rand_idx = np.random.randint(0, N)
        sample_query = names[rand_idx]
        print(f"💡 No query specified. Running random query demo on: '{sample_query}'")
        run_local_search(db_path=db_path, query_name=sample_query, mode=mode, top_k=top_k)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local SABER Embedding Search CLI")
    parser.add_argument("--db", type=str, default="saber_search_db.pth", help="Path to saber_search_db.pth")
    parser.add_argument("--query", type=str, default=None, help="Query image name (e.g. 'patch_0_0')")
    parser.add_argument("--tag", type=str, default=None, help="Land-cover class tag (e.g. 'Water', 'Forest')")
    parser.add_argument("--mode", type=str, default="cross", choices=["cross", "same", "hybrid"], help="Search mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to return")
    args = parser.parse_args()

    run_local_search(db_path=args.db, query_name=args.query, query_class=args.tag, mode=args.mode, top_k=args.top_k)
