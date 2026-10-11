# src/pipeline.py  -- run with:  python -m src.pipeline [--reindex]
import os
import sys
from src.config import *
from src.indexing.build_index import main as phase1
from src.client.client_query import main as phase2
from src.cloud.cloud_retrieval import main as phase3
from src.verification.verify_results import main as phase4

def main():
    print("=== Phase 5: Full Integration (Phases 1-4) ===")
    # Re-embedding the whole corpus on every query was the slowest part of the
    # CLI; only index when asked or when no index exists yet.
    if "--reindex" in sys.argv or not os.path.exists(os.path.join(CLUSTER_DIR, "centroids.npy")):
        print("\nRunning Phase 1...")
        phase1()
    else:
        print("\nPhase 1 skipped (index exists; pass --reindex to rebuild).")
    print("\nRunning Phase 2...")
    query, selected, masked_query = phase2()
    print("\nRunning Phase 3...")
    results = phase3(selected, masked_query)
    print("\nRunning Phase 4...")
    answer = phase4(query, results)
    print("\n=== Final Answer ===")
    print(answer)

if __name__ == "__main__":
    main()