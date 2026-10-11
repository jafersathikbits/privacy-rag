# src/verification/verify_results.py
import ollama
from src.config import *
from src.utils import *

def main(query, retrieved_results):
    print("=== Phase 4: Verification & LLM ===")
    verified_chunks = []

    for cluster_id, idx_in_cluster, chunk_text, proof, root_hash, dist in retrieved_results:
        if verify_merkle_proof(chunk_text, proof, root_hash):
            print(f"[OK] Chunk from cluster {cluster_id} verified.")
            verified_chunks.append(chunk_text)
        else:
            print(f"[FAIL] Chunk from cluster {cluster_id} failed verification.")

    if not verified_chunks:
        print("No verified chunks. Cannot generate answer.")
        return "No verified chunks found."

    # Results arrive best-first; cap what we send so it fits the LLM window.
    context = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(verified_chunks[:CONTEXT_CHUNKS], 1))

    response = ollama.chat(model=LLM_MODEL, messages=[
        {'role': 'system', 'content': "Answer the user question based solely on the provided context. "
                                      "Cite passages like [1]. If the answer is not in the context, say you don't know."},
        {'role': 'user', 'content': f"Context:\n{context}\n\nQuestion: {query}"}
    ], options={"num_ctx": LLM_NUM_CTX, "num_predict": LLM_MAX_TOKENS})
    answer = response['message']['content']
    print(f"Answer:\n{answer}")
    return answer

if __name__ == "__main__":
    print("This script is meant to be called from src/pipeline.py")