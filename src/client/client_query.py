# src/client/client_query.py
import os
import sys
import numpy as np
from src.config import *
from src.utils import *
from src.client.query_processor import preprocess_query

def main():
    print("=== Phase 2: Client Query ===")
    perm = np.load(os.path.join(TRANSFORM_DIR, "perm.npy"))
    signs = np.load(os.path.join(TRANSFORM_DIR, "signs.npy"))
    centroids = np.load(os.path.join(CLUSTER_DIR, "centroids.npy"))

    raw_query = input("Enter your query: ").strip()
    if not raw_query:
        print("Empty query. Exiting.")
        sys.exit(1)

    query = preprocess_query(raw_query, use_llm_rewrite=USE_LLM_REWRITE)
    print(f"Processed query: {query}")

    model = get_embedding_model()
    query_emb = embed_query(query, model)
    masked_query = apply_transform(query_emb, perm, signs)
    masked_query = normalize_vectors(masked_query)

    norm_centroids = normalize_vectors(centroids)
    similarities = np.dot(norm_centroids, masked_query)

    # --- NEW CLUSTER SELECTION: top N ---
    top_idx = np.argsort(similarities)[::-1]
    selected = top_idx[:NUM_CLUSTERS_TO_SELECT].tolist()

    print(f"Selected clusters: {selected}")
    return query, selected, masked_query.tolist()

if __name__ == "__main__":
    main()