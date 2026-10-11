# src/utils/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.config import *
from src.utils import embed_texts, apply_transform

def main():
    print("=== Visualisation of Embedding Masking ===")
    # Load a sample chunk
    import pickle
    with open(os.path.join(CLUSTER_DIR, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    sample_chunk = chunks[0]

    # Load transform
    perm = np.load(os.path.join(TRANSFORM_DIR, "perm.npy"))
    signs = np.load(os.path.join(TRANSFORM_DIR, "signs.npy"))

    raw_emb = embed_texts([sample_chunk], show_progress_bar=False)[0]
    masked_emb = apply_transform(raw_emb, perm, signs)

    # PCA to 2D
    pca = PCA(n_components=2)
    combined = np.vstack([raw_emb, masked_emb])
    pca_result = pca.fit_transform(combined)
    raw_2d = pca_result[0]
    masked_2d = pca_result[1]

    plt.figure(figsize=(8,6))
    plt.scatter(raw_2d[0], raw_2d[1], color='blue', label='Raw Embedding', s=100)
    plt.scatter(masked_2d[0], masked_2d[1], color='red', label='Masked Embedding', s=100)
    plt.arrow(raw_2d[0], raw_2d[1], masked_2d[0]-raw_2d[0], masked_2d[1]-raw_2d[1],
              head_width=0.05, head_length=0.1, fc='gray', ec='gray')
    plt.title('Raw vs Masked Embedding (PCA projection)')
    plt.legend()
    plt.grid(True)
    plt.savefig('embedding_masking.png')
    print("Saved embedding_masking.png")

    print("\n=== Merkle Root Hashes ===")
    actual_n_clusters_path = os.path.join(CLUSTER_DIR, "actual_n_clusters.npy")
    n_clusters = int(np.load(actual_n_clusters_path)) if os.path.exists(actual_n_clusters_path) else N_CLUSTERS_MAX
    for cluster_id in range(n_clusters):
        root_path = os.path.join(MERKLE_DIR, f"cluster_{cluster_id}_root.txt")
        if os.path.exists(root_path):
            with open(root_path, "r") as f:
                print(f"Cluster {cluster_id}: {f.read().strip()[:20]}...")

    print("\nVisualisation complete. Open 'embedding_masking.png' to see the effect.")

if __name__ == "__main__":
    main()