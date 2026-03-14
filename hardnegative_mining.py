"""
Hard-negative mining pipeline

Features:
- CodeBERT embeddings (GPU if available)
- FAISS exact cosine similarity search (GPU if faiss-gpu installed)
- Cosine similarity threshold = 0.90
- top_k = 50 neighbors fetched per sample
- Vectorized opposite-class selection (no Python loops)
- Atomic caching (.pt preferred, .npy fallback); auto-recover corrupted files
- Batched FAISS search to control memory
- Similarity score saved in output CSV
- Resume-safe (skips recomputation if cache exists)
"""

import argparse
import os
import sys
import time
import math
import gc
import tempfile
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Hard-negative mining pipeline")
    parser.add_argument("--csv-path", help="path to input CSV")
    parser.add_argument("--output-dir", help="output directory")
    parser.add_argument("--top-k", type=int, default=5, help="neighbors fetched per sample")
    parser.add_argument("--batch-size", type=int, default=32, help="CodeBERT encoding batch size")
    parser.add_argument("--search-batch", type=int, default=100_000, help="FAISS search batch size")
    parser.add_argument("--text-col", default="function_before", help="text column in CSV for encoding")
    parser.add_argument("--label-cands", nargs="+", default=("is_vul", "target", "label"), help="candidate label columns")
    parser.add_argument("--similarity-threshold", type=float, default=0.90, help="cosine similarity threshold")
    parser.add_argument("--max-delete-retry", type=int, default=3, help="retry attempts for filesystem delete")
    return parser.parse_args()

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load CodeBERT
# =========================
print("\n[INFO] Loading CodeBERT tokenizer & model …")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    trust_remote_code=True,
    use_safetensors=True,
).to(device)
model.eval()
print("[INFO] Model ready.\n")

# =========================
# Atomic save helpers
# =========================

def atomic_save_tensor(t: torch.Tensor, path: str) -> None:
    """Save tensor to temp file then rename → prevents half-written files on crash."""
    dirpath = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(delete=False, dir=dirpath) as f:
        tmpname = f.name
    torch.save(t, tmpname)
    os.replace(tmpname, path)


def atomic_save_numpy(arr: np.ndarray, path: str) -> None:
    """Save numpy array to temp file then rename."""
    dirpath = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(delete=False, dir=dirpath, suffix=".npy") as f:
        tmpname = f.name
    np.save(tmpname, arr)
    os.replace(tmpname, path)


def safe_remove(path: str, max_delete_retry: int = 3) -> None:
    """Delete file with retries to handle Windows file-locking."""
    if not os.path.exists(path):
        return
    for attempt in range(max_delete_retry):
        try:
            os.remove(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.5 + attempt * 0.5)
    os.remove(path)  # final attempt — raise if still locked

# =========================
# Robust cache loader
# =========================

def load_cached_embeddings(pt_path: str, npy_path: str, max_delete_retry: int):
    """
    Try loading embeddings from disk.
    Returns CPU torch.Tensor or None (signals recompute needed).
    """
    # --- try .pt first ---
    if os.path.exists(pt_path):
        try:
            print(f"🔄 Loading cached embeddings: {pt_path}")
            t = torch.load(pt_path, map_location="cpu")
            if not isinstance(t, torch.Tensor):
                raise ValueError("File is not a torch.Tensor")
            return t
        except Exception as e:
            print(f"⚠️  Corrupted .pt ({e}) → deleting and recomputing.")
            try:
                safe_remove(pt_path, max_delete_retry=max_delete_retry)
            except Exception:
                pass

    # --- fallback .npy ---
    if os.path.exists(npy_path):
        print(f"🔄 Loading cached embeddings: {npy_path}")
        arr = None
        try:
            arr = np.load(npy_path, allow_pickle=False)
            if isinstance(arr, np.lib.npyio.NpzFile):
                arr.close()
                raise ValueError("Got .npz archive instead of .npy array.")
            return torch.from_numpy(np.asarray(arr, dtype=np.float32))
        except Exception as e:
            print(f"⚠️  Corrupted .npy ({e}) → deleting and recomputing.")
            try:
                if hasattr(arr, "close"):
                    arr.close()
                safe_remove(npy_path, max_delete_retry=max_delete_retry)
            except Exception as e2:
                print(f"❌ Could not delete {npy_path}: {e2}")

    return None

# =========================
# Label normalization
# =========================

def normalize_labels_binary(series: pd.Series) -> pd.Series:
    """Normalize any label encoding (bool / string / numeric) to integer 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        mapped = series.astype(str).str.lower().map({"true": 1, "false": 0})
        if mapped.isna().any():
            mapped = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        return mapped.clip(0, 1)
    return series.fillna(0).astype(int).clip(0, 1)

# =========================
# Embedding computation
# =========================

def compute_embeddings(df: pd.DataFrame, text_col: str, output_dir: str, batch_size: int, max_delete_retry: int) -> torch.Tensor:
    """
    Encode all rows in df[text_col] with CodeBERT (CLS token).
    Returns a CPU float32 tensor of shape (n, 768).
    Caches to output_dir/vulgate_embeddings.pt and .npy.
    """
    pt_path  = os.path.join(output_dir, "vulgate_embeddings.pt")
    npy_path = os.path.join(output_dir, "vulgate_embeddings.npy")

    cached = load_cached_embeddings(pt_path, npy_path, max_delete_retry)
    if cached is not None:
        print(f"   Shape: {list(cached.shape)}")
        return cached

    texts = df[text_col].astype(str).tolist()
    n = len(texts)
    print(f"⚡ Encoding {n} code functions with CodeBERT …")

    collected = []
    with torch.no_grad():
        for i in tqdm(
            range(0, n, batch_size),
            total=math.ceil(n / batch_size),
            desc="🔨 Embedding",
            dynamic_ncols=True,
        ):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            # CLS token at position 0 = summary embedding of the whole function
            emb = outputs.last_hidden_state[:, 0, :].detach().cpu()
            collected.append(emb)

    all_embeddings = torch.cat(collected, dim=0)   # (n, 768) on CPU
    print(f"   Embedding shape: {list(all_embeddings.shape)}")

    # save atomically in both formats
    try:
        atomic_save_tensor(all_embeddings, pt_path)
        print(f"💾 Saved → {pt_path}")
    except Exception as e:
        print(f"⚠️  Could not save .pt: {e}")
    try:
        atomic_save_numpy(all_embeddings.numpy(), npy_path)
        print(f"💾 Saved → {npy_path}")
    except Exception as e:
        print(f"⚠️  Could not save .npy: {e}")

    return all_embeddings

# =========================
# FAISS index builder
# =========================

def build_faiss_index(X: np.ndarray):
    """
    L2-normalize X in-place then build IndexFlatIP.
    After normalization: inner product == cosine similarity.
    Returns (index, is_gpu: bool).
    """
    print("🔧 Building FAISS index …")
    faiss.normalize_L2(X)          # unit-normalize → cosine sim = dot product
    dim = X.shape[1]
    cpu_index = faiss.IndexFlatIP(dim)

    try:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            print(f"✅ FAISS GPU detected ({num_gpus} GPU(s)). Moving index to GPU 0.")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            gpu_index.add(X)
            return gpu_index, True
    except Exception as e:
        print(f"ℹ️  FAISS-GPU not available ({e}). Using CPU FAISS.")

    cpu_index.add(X)
    print(f"✅ FAISS CPU index built. Vectors: {cpu_index.ntotal:,}")
    return cpu_index, False

# =========================
# Hard negative mining
# =========================

def mine_hard_negatives(
    df: pd.DataFrame,
    embeddings: torch.Tensor,
    label_col: str,
    output_dir: str,
    top_k: int,
    search_batch: int,
    text_col: str,
    similarity_threshold: float,
    max_delete_retry: int,
) -> pd.DataFrame:
    """
    For every anchor, find the nearest neighbor that:
      1. Has the OPPOSITE label        (hard negative condition)
      2. Has cosine similarity >= threshold (quality condition)

    Output CSV columns:
        anchor, anchor_label, hard_negative, hard_negative_label, similarity
    """
    out_csv = os.path.join(output_dir, "vulgate_hard_negatives.csv")

    if os.path.exists(out_csv):
        print(f"🔄 Loading cached results: {out_csv}")
        return pd.read_csv(out_csv)

    n = len(df)
    print(f"\n⚡ Mining hard negatives …")
    print(f"   Samples   : {n:,}")
    print(f"   k         : {top_k}")
    print(f"   Threshold : {similarity_threshold}")

    # convert to contiguous float32 numpy
    X = embeddings.detach().cpu().numpy().astype(np.float32, order="C")

    # build FAISS cosine index
    index, is_gpu = build_faiss_index(X)

    # --- batched search — stores neighbor indices AND cosine scores ---
    nbrs_all   = np.empty((n, top_k + 1), dtype=np.int64)
    scores_all = np.empty((n, top_k + 1), dtype=np.float32)

    for st in tqdm(
        range(0, n, search_batch),
        total=math.ceil(n / search_batch),
        desc="🔎 FAISS search",
        dynamic_ncols=True,
    ):
        en = min(st + search_batch, n)
        D, I = index.search(X[st:en], top_k + 1)
        nbrs_all[st:en]   = I
        scores_all[st:en] = D   # cosine similarity scores

    # --- prepare label and text arrays ---
    labels = normalize_labels_binary(df[label_col]).to_numpy().astype(np.int64)
    texts  = df[text_col].astype(str).to_numpy()

    # drop column 0 (self-match, sim ≈ 1.0)
    nbrs   = nbrs_all[:, 1:]     # (n, top_k)
    scores = scores_all[:, 1:]   # (n, top_k)  cosine similarity values

    # neighbor labels via fancy indexing
    nbr_labels = labels[nbrs]    # (n, top_k)

    # --- dual condition mask ---
    opposite_mask  = (nbr_labels != labels[:, None])   # opposite class
    threshold_mask = (scores >= similarity_threshold)  # above cosine threshold
    valid_mask     = opposite_mask & threshold_mask    # BOTH must be True

    has_any = valid_mask.any(axis=1)

    # --- print filter statistics ---
    qualified = int(has_any.sum())
    rejected  = n - qualified
    pct_kept  = 100.0 * qualified / max(n, 1)

    print(f"\n📊 Threshold filter results (cosine ≥ {similarity_threshold}):")
    print(f"   Total anchors        : {n:,}")
    print(f"   Qualified pairs      : {qualified:,}  ({pct_kept:.1f}%)")
    print(f"   Rejected (below thr) : {rejected:,}  ({100 - pct_kept:.1f}%)")

    if not has_any.any():
        print(f"\n⚠️  No hard negatives found above threshold {similarity_threshold}.")
        print(f"   Tip: lower --similarity-threshold and re-run.")
        empty_df = pd.DataFrame(
            columns=["anchor", "anchor_label", "hard_negative", "hard_negative_label", "similarity"]
        )
        empty_df.to_csv(out_csv, index=False)
        return empty_df

    # argmax → index of first qualifying neighbor per row
    first_pos       = np.argmax(valid_mask, axis=1)          # (n,)
    keep_idx        = np.where(has_any)[0]
    chosen_nbr_idx  = nbrs[np.arange(n), first_pos]          # global indices
    chosen_scores   = scores[np.arange(n), first_pos]        # cosine sim values

    out_df = pd.DataFrame({
        "anchor"             : texts[keep_idx],
        "anchor_label"       : labels[keep_idx].astype(int),
        "hard_negative"      : texts[chosen_nbr_idx[keep_idx]],
        "hard_negative_label": labels[chosen_nbr_idx[keep_idx]].astype(int),
        "similarity"         : chosen_scores[keep_idx].round(6),
    })

    out_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved {len(out_df):,} hard negative pairs → {out_csv}")

    # similarity distribution of accepted pairs
    sim_vals = out_df["similarity"]
    print(f"\n   Similarity stats of accepted pairs:")
    print(f"   min    = {sim_vals.min():.4f}")
    print(f"   mean   = {sim_vals.mean():.4f}")
    print(f"   median = {sim_vals.median():.4f}")
    print(f"   max    = {sim_vals.max():.4f}")

    # label distribution
    print(f"\n   Anchor label distribution:")
    print(f"   {out_df['anchor_label'].value_counts().to_dict()}")

    return out_df

# =========================
# Main
# =========================

def main():

    args = parse_args()
    csv_path = args.csv_path
    output_dir = args.output_dir
    top_k = args.top_k
    batch_size = args.batch_size
    search_batch = args.search_batch
    text_col = args.text_col
    label_cands = tuple(args.label_cands)
    similarity_threshold = args.similarity_threshold
    max_delete_retry = args.max_delete_retry

    print(f"[INFO] Torch device        : {device}")
    print(f"[INFO] Similarity threshold: {similarity_threshold}")
    print(f"[INFO] Input CSV           : {csv_path}")
    os.makedirs(output_dir, exist_ok=True)

    # --- load CSV ---
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        print(f"   Make sure '{csv_path}' is in the same directory as this script.")
        sys.exit(1)

    print(f"📂 Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {list(df.columns)}")

    # --- detect label column ---
    label_col = next((c for c in label_cands if c in df.columns), None)
    if label_col is None:
        print(f"\n❌ No label column found. Looked for: {label_cands}")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Add your label column name to label_cands in the CONFIG section.")
        sys.exit(1)

    print(f"   Label column: '{label_col}'")
    df[label_col] = normalize_labels_binary(df[label_col])
    print(f"   Label distribution: {df[label_col].value_counts().to_dict()}")

    # --- check text column ---
    if text_col not in df.columns:
        print(f"\n❌ Text column '{text_col}' not found in CSV.")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Update text_col in the CONFIG section.")
        sys.exit(1)

    # --- step 1: embed ---
    print("\n--- Step 1: Compute Embeddings ---")
    embeddings = compute_embeddings(df, text_col, output_dir, batch_size, max_delete_retry)

    # --- step 2: mine ---
    print("\n--- Step 2: Mine Hard Negatives ---")
    hn_df = mine_hard_negatives(
        df,
        embeddings,
        label_col,
        output_dir,
        top_k,
        search_batch,
        text_col,
        similarity_threshold,
        max_delete_retry,
    )

    # --- done ---
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Hard negatives : {len(hn_df):,}")
    print(f"  Output folder  : {os.path.abspath(output_dir)}/")
    print(f"    ├── vulgate_embeddings.pt")
    print(f"    ├── vulgate_embeddings.npy")
    print(f"    └── vulgate_hard_negatives.csv")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()