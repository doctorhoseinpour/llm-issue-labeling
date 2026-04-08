#!/usr/bin/env python3
"""
build_and_query_index.py
========================
Build a FAISS index from the FULL dataset (retrieval pool), then select test
issues as the first 50% of each label (stratified, no shuffle, preserving
original order). Retrieve neighbors for test issues only, with strict
self-match exclusion.

Self-match exclusion uses TWO mechanisms:
  1. Corpus index exclusion: skip the test issue's own row in the index
  2. Content hash exclusion: skip any row with identical title+body

Output CSV columns:
  test_idx, test_title, test_body, test_label, [test_created_at],
  neighbor_rank, neighbor_title, neighbor_body, neighbor_label

Usage:
  python build_and_query_index.py \
    --dataset data/issues3k.csv \
    --top_ks 1,3,9,15 \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --output_dir results/neighbors
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
_whitespace = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace, strip."""
    if not text:
        return ""
    return _whitespace.sub(" ", str(text)).strip()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dedup_key(row) -> str:
    t = str(row.get("title", "") or "").strip().lower()
    b = str(row.get("body", "") or "").strip().lower()
    return hashlib.md5(f"{t}||{b}".encode("utf-8")).hexdigest()


def deduplicate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Drop exact title+body duplicates, report count."""
    if df.empty:
        return df
    keys = df.apply(_dedup_key, axis=1)
    mask = ~keys.duplicated(keep="first")
    removed = (~mask).sum()
    if removed:
        print(f"  Removed {removed} duplicate issues from {name} dataset.")
    else:
        print(f"  No duplicates in {name} dataset.")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Normalise label column
# ---------------------------------------------------------------------------

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a 'labels' column exists."""
    if "labels" in df.columns:
        return df
    if "label" in df.columns:
        df = df.rename(columns={"label": "labels"})
        return df
    raise ValueError(f"CSV needs a 'labels' or 'label' column. Found: {list(df.columns)}")


# ---------------------------------------------------------------------------
# Stratified test split: first 50% of each label, no shuffle
# ---------------------------------------------------------------------------

def select_test_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the first 50% of rows for each label as test issues.
    Preserves original row order within each label group.
    Returns a DataFrame with the same columns plus '__corpus_idx' = original
    position in the full (deduplicated) dataset.
    """
    df = df.copy()
    df["__corpus_idx"] = df.index  # track position in full corpus

    label_col = "labels"
    labels = sorted(df[label_col].unique())

    test_parts = []
    for lab in labels:
        group = df[df[label_col] == lab]
        n_test = len(group) // 2
        test_parts.append(group.iloc[:n_test])

    test_df = pd.concat(test_parts, ignore_index=False)
    # Sort by original corpus index to maintain overall dataset order
    test_df = test_df.sort_values("__corpus_idx").reset_index(drop=True)

    print(f"  Test split (first 50% per label, no shuffle):")
    for lab in labels:
        total = (df[label_col] == lab).sum()
        selected = (test_df[label_col] == lab).sum()
        print(f"    {lab}: {selected}/{total}")
    print(f"  Total test issues: {len(test_df)} / {len(df)}")

    return test_df


# ---------------------------------------------------------------------------
# FAISS index build + query
# ---------------------------------------------------------------------------

def build_faiss_index(texts: List[str], embeddings_model):
    """Embed all texts and build a flat FAISS inner-product index."""
    import faiss

    print(f"  Embedding {len(texts)} documents...")
    t0 = time.time()
    vectors = embeddings_model.embed_documents(texts)
    vectors = np.array(vectors, dtype="float32")
    elapsed = time.time() - t0
    print(f"  Embedded in {elapsed:.1f}s  (shape={vectors.shape})")

    # Normalise for cosine similarity (inner product on unit vectors = cosine)
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"  FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index, vectors


def query_index(
    test_corpus_indices: List[int],
    test_dedup_keys: List[str],
    corpus_dedup_keys: List[str],
    embeddings_model,
    index,
    corpus_texts: List[str],
    test_texts: List[str],
    max_k: int,
) -> List[List[int]]:
    """
    Query the FAISS index for each test issue. Returns list-of-lists of
    corpus indices (length = max_k per test issue).

    Self-match exclusion (two layers):
      1. Skip if corpus index == test issue's own corpus index
      2. Skip if corpus dedup key == test issue's dedup key (catches
         any remaining content-identical rows)
    """
    import faiss

    print(f"  Querying index for {len(test_texts)} test issues (max_k={max_k})...")
    t0 = time.time()

    # Embed test issues
    test_vecs = np.array(embeddings_model.embed_documents(test_texts), dtype="float32")
    faiss.normalize_L2(test_vecs)

    # Over-fetch to allow skipping self-matches and duplicates
    fetch_k = min(max_k + 50, index.ntotal)
    distances, indices = index.search(test_vecs, fetch_k)

    all_neighbors: List[List[int]] = []
    self_match_count = 0

    for qi in range(len(test_texts)):
        own_corpus_idx = test_corpus_indices[qi]
        own_key = test_dedup_keys[qi]
        neighbors = []
        for j in range(fetch_k):
            ci = int(indices[qi, j])
            if ci < 0:
                continue
            # Layer 1: exact index exclusion
            if ci == own_corpus_idx:
                self_match_count += 1
                continue
            # Layer 2: content hash exclusion
            if corpus_dedup_keys[ci] == own_key:
                self_match_count += 1
                continue
            neighbors.append(ci)
            if len(neighbors) >= max_k:
                break
        all_neighbors.append(neighbors)

    elapsed = time.time() - t0
    print(f"  Querying done in {elapsed:.1f}s")
    print(f"  Self-matches excluded: {self_match_count}")
    return all_neighbors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and retrieve neighbors.")
    parser.add_argument("--dataset", required=True,
                        help="Full dataset CSV (used as both retrieval pool and source of test issues)")
    parser.add_argument("--top_ks", default="1,3,9,15", help="Comma-separated k values")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", required=True, help="Directory for output CSVs")
    parser.add_argument("--cache_dir", default=".faiss_cache", help="Directory to cache index")
    parser.add_argument("--model_cache_dir", default=None,
                        help="Directory to download/cache HuggingFace models (default: HF default cache)")
    args = parser.parse_args()

    ks = sorted(set(int(x) for x in args.top_ks.split(",")))
    max_k = max(ks)
    print(f"K values: {ks}  (max_k={max_k})")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load and deduplicate full dataset ---
    print("Loading data...")
    full_df = ensure_labels(pd.read_csv(args.dataset))
    full_df["body"] = full_df["body"].fillna("")
    full_df["title"] = full_df["title"].fillna("")
    full_df["labels"] = full_df["labels"].astype(str).str.lower().str.strip()
    full_df = deduplicate(full_df, "full")
    print(f"  Full corpus: {len(full_df)} issues")

    # --- Select test issues (first 50% of each label) ---
    test_df = select_test_issues(full_df)
    test_corpus_indices = test_df["__corpus_idx"].tolist()

    # --- Prepare texts and dedup keys ---
    corpus_texts = (full_df["title"] + " " + full_df["body"]).apply(clean_text).tolist()
    test_texts = (test_df["title"] + " " + test_df["body"]).apply(clean_text).tolist()
    corpus_keys = full_df.apply(_dedup_key, axis=1).tolist()
    test_keys = [corpus_keys[ci] for ci in test_corpus_indices]

    # --- Load embedding model ---
    print(f"Loading embedding model: {args.embedding_model}")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embed_kwargs = {"model_name": args.embedding_model, "model_kwargs": {"device": "cuda"}}
    if args.model_cache_dir:
        embed_kwargs["cache_folder"] = args.model_cache_dir
    embed_model = HuggingFaceEmbeddings(**embed_kwargs)

    # --- Build or load index (from FULL corpus) ---
    import faiss
    os.makedirs(args.cache_dir, exist_ok=True)
    safe_name = os.path.basename(args.dataset) + "_" + args.embedding_model.replace("/", "_")
    cache_hash = hashlib.md5(open(args.dataset, "rb").read()).hexdigest()[:12]
    cache_path = os.path.join(args.cache_dir, f"{safe_name}_{cache_hash}")
    index_file = cache_path + ".index"

    if os.path.exists(index_file):
        print(f"Loading cached FAISS index from {index_file}")
        index = faiss.read_index(index_file)
    else:
        print("Building FAISS index from full corpus...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        index, vectors = build_faiss_index(corpus_texts, embed_model)
        faiss.write_index(index, index_file)
        print(f"  Cached index to {index_file}")
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  GPU peak memory during indexing: {peak:.0f} MB")

    # --- Query (test issues only, with self-exclusion) ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    all_neighbors = query_index(
        test_corpus_indices=test_corpus_indices,
        test_dedup_keys=test_keys,
        corpus_dedup_keys=corpus_keys,
        embeddings_model=embed_model,
        index=index,
        corpus_texts=corpus_texts,
        test_texts=test_texts,
        max_k=max_k,
    )
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  GPU peak memory during retrieval: {peak:.0f} MB")

    # --- Verify no self-retrieval ---
    violations = 0
    for qi in range(len(test_df)):
        own_ci = test_corpus_indices[qi]
        own_key = test_keys[qi]
        for ni in all_neighbors[qi]:
            if ni == own_ci or corpus_keys[ni] == own_key:
                violations += 1
    if violations:
        print(f"  WARNING: {violations} self-retrieval violations found!")
    else:
        print(f"  Verified: 0 self-retrieval violations across all test issues.")

    # --- Write output CSVs (one per k) ---
    has_created = "created_at" in test_df.columns

    for k in ks:
        rows = []
        for qi in range(len(test_df)):
            test_row = test_df.iloc[qi]
            neighbors = all_neighbors[qi][:k]
            for rank, ci in enumerate(neighbors):
                corpus_row = full_df.iloc[ci]
                row = {
                    "test_idx": qi,
                    "test_title": test_row["title"],
                    "test_body": test_row["body"],
                    "test_label": test_row["labels"],
                }
                if has_created:
                    row["test_created_at"] = test_row.get("created_at", "")
                row.update({
                    "neighbor_rank": rank,
                    "neighbor_title": corpus_row["title"],
                    "neighbor_body": corpus_row["body"],
                    "neighbor_label": corpus_row["labels"],
                })
                rows.append(row)

        out_path = os.path.join(args.output_dir, f"neighbors_k{k}.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  Wrote {out_path}  ({len(rows)} rows for {len(test_df)} test issues, k={k})")

    # --- Save test split metadata for reference ---
    meta_path = os.path.join(args.output_dir, "test_split_info.csv")
    test_meta = test_df[["__corpus_idx", "title", "labels"]].copy()
    test_meta.index.name = "test_idx"
    test_meta.to_csv(meta_path)
    print(f"  Wrote test split metadata to {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()