# src/cloud/cloud_retrieval.py
import os
import pickle
import numpy as np
import faiss
from src.config import *
from src.utils import *

def main(selected_clusters, masked_query, top_k=TOP_K):
    print("=== Phase 3: Cloud Retrieval ===")
    # normalize_vectors returns a copy; the result was previously discarded.
    masked_query = normalize_vectors(np.array(masked_query, dtype=np.float32).reshape(1, -1))

    all_results = []
    for cluster_id in selected_clusters:
        index_path = os.path.join(FAISS_DIR, f"cluster_{cluster_id}.index")
        if not os.path.exists(index_path):
            print(f"Cluster {cluster_id} index not found, skipping.")
            continue

        index = faiss.read_index(index_path)
        distances, indices = index.search(masked_query, min(top_k, index.ntotal))

        chunks_path = os.path.join(CLUSTER_DIR, f"cluster_{cluster_id}_chunks.pkl")
        proofs_path = os.path.join(MERKLE_DIR, f"cluster_{cluster_id}_proofs.pkl")
        root_path = os.path.join(MERKLE_DIR, f"cluster_{cluster_id}_root.txt")
        if not (os.path.exists(chunks_path) and os.path.exists(proofs_path) and os.path.exists(root_path)):
            print(f"Missing metadata for cluster {cluster_id}, skipping.")
            continue

        with open(chunks_path, "rb") as f:
            cluster_chunks = pickle.load(f)
        with open(proofs_path, "rb") as f:
            cluster_proofs = pickle.load(f)
        with open(root_path, "r") as f:
            root_hash_hex = f.read().strip()
            root_hash = bytes.fromhex(root_hash_hex)

        for idx_in_cluster, dist in zip(indices[0], distances[0]):
            if idx_in_cluster == -1:
                continue
            chunk_text = cluster_chunks[idx_in_cluster]
            proof = cluster_proofs[idx_in_cluster]
            all_results.append((cluster_id, idx_in_cluster, chunk_text, proof, root_hash, dist))

    all_results.sort(key=lambda x: x[5], reverse=True)
    print(f"Retrieved {len(all_results)} chunks.")
    return all_results[:top_k]

if __name__ == "__main__":
    print("This script is meant to be called from src/pipeline.py")