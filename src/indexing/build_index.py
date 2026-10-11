# src/indexing/build_index.py
import glob
import hashlib
import math
import os
import pickle
import numpy as np
import faiss
from sklearn.cluster import KMeans
from tqdm import tqdm
from src.config import *
from src.utils import *


def _list_documents():
    file_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for file in sorted(files):
            if os.path.splitext(file)[1].lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(os.path.join(root, file))
    return file_paths


def _transform_exists():
    return (os.path.exists(os.path.join(TRANSFORM_DIR, "perm.npy"))
            and os.path.exists(os.path.join(TRANSFORM_DIR, "signs.npy")))


def _load_or_create_transform(dim):
    """Reuse the existing secret transform so cached masked embeddings stay valid."""
    perm_path = os.path.join(TRANSFORM_DIR, "perm.npy")
    signs_path = os.path.join(TRANSFORM_DIR, "signs.npy")
    if os.path.exists(perm_path) and os.path.exists(signs_path):
        perm, signs = np.load(perm_path), np.load(signs_path)
        if perm.shape[0] == dim:
            return perm, signs, False
    perm, signs = generate_transform(dim)
    np.save(perm_path, perm)
    np.save(signs_path, signs)
    return perm, signs, True


def _clear_dir(pattern):
    for p in glob.glob(pattern):
        try:
            os.remove(p)
        except OSError:
            pass


def _choose_n_clusters(num_samples):
    # One cluster per chunk (the old behaviour for small corpora) makes cluster
    # routing meaningless; ~sqrt(n) is the usual IVF heuristic.
    return max(1, min(N_CLUSTERS_MAX, int(math.sqrt(num_samples))))


def main():
    print("=== Phase 1: Offline Indexing ===")
    file_paths = _list_documents()
    print(f"Found {len(file_paths)} files")

    # The masked-embedding cache is only valid for the current secret transform.
    if not _transform_exists():
        _clear_dir(os.path.join(EMBED_CACHE_DIR, "*.pkl"))

    # ---- Extract + chunk, reusing cached masked embeddings per file ----
    model_tag = hashlib.sha1(EMBEDDING_MODEL.encode()).hexdigest()[:8]
    all_chunks, chunk_sources, emb_parts = [], [], []
    to_embed = []          # (index in emb_parts, chunks, cache_path)
    used_cache_files = set()
    for fp in tqdm(file_paths, desc="Extracting text"):
        digest = file_sha256(fp)
        cache_path = os.path.join(EMBED_CACHE_DIR, f"{digest}_{model_tag}_{CHUNK_SIZE}_{CHUNK_OVERLAP}.pkl")
        used_cache_files.add(os.path.basename(cache_path))
        cached = load_pickle(cache_path) if os.path.exists(cache_path) else None
        if cached is not None:
            chunks = cached["chunks"]
        else:
            text = extract_text_from_file(fp)
            chunks = chunk_text(text) if text else []
        if not chunks:
            continue
        all_chunks.extend(chunks)
        chunk_sources.extend([os.path.basename(fp)] * len(chunks))
        emb_parts.append(cached["embeddings"] if cached is not None else None)
        if cached is None:
            to_embed.append((len(emb_parts) - 1, chunks, cache_path))

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        print("No chunks extracted. Clearing old index.")
        for d in (CLUSTER_DIR, FAISS_DIR, MERKLE_DIR):
            _clear_dir(os.path.join(d, "*"))
        return

    # ---- Embed only new / changed files ----
    if to_embed:
        new_texts = [c for _, chunks, _ in to_embed for c in chunks]
        print(f"Embedding {len(new_texts)} new chunks ({len(all_chunks) - len(new_texts)} cached)...")
        new_raw = embed_texts(new_texts, BATCH_SIZE)
        perm, signs, regenerated = _load_or_create_transform(new_raw.shape[1])
        if regenerated and len(to_embed) != len(emb_parts):
            # Embedding dimension changed under a cached corpus; start clean.
            _clear_dir(os.path.join(EMBED_CACHE_DIR, "*.pkl"))
            return main()
        new_masked = apply_transform(new_raw, perm, signs).astype(np.float32)
        del new_raw   # raw embeddings are never written to disk
        offset = 0
        for part_idx, chunks, cache_path in to_embed:
            part = new_masked[offset:offset + len(chunks)]
            offset += len(chunks)
            emb_parts[part_idx] = part
            save_pickle({"chunks": chunks, "embeddings": part}, cache_path)
    else:
        print("All embeddings loaded from cache.")

    # Drop cache entries for files that no longer exist
    for p in glob.glob(os.path.join(EMBED_CACHE_DIR, "*.pkl")):
        if os.path.basename(p) not in used_cache_files:
            os.remove(p)

    transformed_emb = normalize_vectors(np.vstack(emb_parts))
    dim = transformed_emb.shape[1]
    print(f"Masked embeddings shape: {transformed_emb.shape}")

    # ---- Clustering ----
    num_samples = len(transformed_emb)
    actual_n_clusters = _choose_n_clusters(num_samples)
    if actual_n_clusters == 1:
        labels = np.zeros(num_samples, dtype=int)
        centroids = transformed_emb.mean(axis=0, keepdims=True)
        print("Using a single cluster.")
    else:
        print(f"Clustering {num_samples} chunks into {actual_n_clusters} clusters (max {N_CLUSTERS_MAX})...")
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(transformed_emb)
        centroids = kmeans.cluster_centers_.astype(np.float32)

    # Remove files from a previous (possibly larger) index
    for d in (CLUSTER_DIR, FAISS_DIR, MERKLE_DIR):
        _clear_dir(os.path.join(d, "cluster_*"))

    np.save(os.path.join(CLUSTER_DIR, "actual_n_clusters.npy"), actual_n_clusters)
    np.save(os.path.join(CLUSTER_DIR, "centroids.npy"), centroids)
    np.save(os.path.join(CLUSTER_DIR, "labels.npy"), labels)
    save_pickle(all_chunks, os.path.join(CLUSTER_DIR, "chunks.pkl"))
    save_pickle(chunk_sources, os.path.join(CLUSTER_DIR, "chunk_sources.pkl"))
    np.save(os.path.join(CLUSTER_DIR, "transformed_embeddings.npy"), transformed_emb)

    print("Building per-cluster FAISS indexes and Merkle trees...")
    for cluster_id in range(actual_n_clusters):
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        cluster_emb = np.ascontiguousarray(transformed_emb[indices])
        cluster_chunks = [all_chunks[i] for i in indices]
        cluster_sources = [chunk_sources[i] for i in indices]

        index = faiss.IndexFlatIP(dim)
        index.add(cluster_emb)
        faiss.write_index(index, os.path.join(FAISS_DIR, f"cluster_{cluster_id}.index"))

        save_pickle(cluster_chunks, os.path.join(CLUSTER_DIR, f"cluster_{cluster_id}_chunks.pkl"))
        save_pickle(cluster_sources, os.path.join(CLUSTER_DIR, f"cluster_{cluster_id}_sources.pkl"))

        root_hash, proofs = build_merkle_tree(cluster_chunks)
        with open(os.path.join(MERKLE_DIR, f"cluster_{cluster_id}_root.txt"), "w") as f:
            f.write(root_hash.hex())
        save_pickle(proofs, os.path.join(MERKLE_DIR, f"cluster_{cluster_id}_proofs.pkl"))

    print("Phase 1 completed successfully.")


if __name__ == "__main__":
    main()
