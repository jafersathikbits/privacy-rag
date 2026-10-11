"""Privacy-preserving, verifiable RAG — web demo.

Run:  python demo.py   (then open http://localhost:8000)
"""
import csv
import json
import math
import os
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache

import numpy as np
import faiss
import ollama
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from src.config import *
from src.utils import *
from src.client.query_processor import generate_keywords
from src.utils.logger import log_query
from src.indexing.build_index import main as index_main

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
VERBOSE = os.environ.get("RAG_VERBOSE", "1") == "1"
PER_CLUSTER_K = 50          # FAISS hits per probed cluster
KEYWORD_TOP_K = 50          # lexical candidates
RERANK_CANDIDATES = 40      # cross-encoder is the expensive step; keep it bounded
USE_RERANK = True
USE_LLM_KEYWORDS = True     # extra LLM call for short factoid queries
SOURCE_BOOST = 0.15         # soft boost for chunks from the most relevant document
HISTORY_TURNS = 4           # previous messages sent to the LLM
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

UI_FILE = os.path.join(BASE_DIR, "ui", "index.html")
CONVERSATIONS_FILE = os.path.join(INDEX_DIR, "conversations.json")

STOPWORDS = {
    'a', 'an', 'the', 'of', 'for', 'on', 'at', 'to', 'in', 'with', 'without', 'and', 'or',
    'but', 'by', 'from', 'as', 'so', 'then', 'what', 'who', 'whom', 'which', 'when', 'where',
    'why', 'how', 'is', 'are', 'was', 'were', 'does', 'did', 'do', 'can', 'could', 'about',
    'this', 'that', 'these', 'those', 'there', 'their', 'his', 'her', 'its', 'into', 'tell',
}
GREETINGS = {"hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon",
             "good evening", "morning", "afternoon", "evening", "yo", "hiya"}


def log(msg):
    if VERBOSE:
        print(msg)


# ------------------------------------------------------------------
# Index state (loaded once into memory; previously pickles were re-read
# from disk for every cluster on every query)
# ------------------------------------------------------------------
class IndexState:
    def __init__(self):
        self.ready = False
        self.perm = self.signs = self.norm_centroids = None
        self.chunks, self.sources, self.chunks_lower = [], [], []
        self.clusters = {}          # cluster_id -> (faiss index, global indices ndarray)
        self.chunk_meta = {}        # global idx -> (proof, root_hash)
        self.source_centroids = {}  # source -> masked unit centroid
        self.n_clusters = 0

    @staticmethod
    def load():
        st = IndexState()
        if not has_documents() or not os.path.exists(os.path.join(CLUSTER_DIR, "centroids.npy")):
            return st
        st.perm = np.load(os.path.join(TRANSFORM_DIR, "perm.npy"))
        st.signs = np.load(os.path.join(TRANSFORM_DIR, "signs.npy"))
        centroids = np.load(os.path.join(CLUSTER_DIR, "centroids.npy"))
        st.norm_centroids = normalize_vectors(centroids)
        n_path = os.path.join(CLUSTER_DIR, "actual_n_clusters.npy")
        st.n_clusters = int(np.load(n_path)) if os.path.exists(n_path) else centroids.shape[0]
        st.chunks = load_pickle(os.path.join(CLUSTER_DIR, "chunks.pkl"))
        st.sources = load_pickle(os.path.join(CLUSTER_DIR, "chunk_sources.pkl"))
        st.chunks_lower = [c.lower() for c in st.chunks]
        labels = np.load(os.path.join(CLUSTER_DIR, "labels.npy"))

        for cid in range(st.n_clusters):
            index_path = os.path.join(FAISS_DIR, f"cluster_{cid}.index")
            proofs_path = os.path.join(MERKLE_DIR, f"cluster_{cid}_proofs.pkl")
            root_path = os.path.join(MERKLE_DIR, f"cluster_{cid}_root.txt")
            if not all(os.path.exists(p) for p in (index_path, proofs_path, root_path)):
                continue
            proofs = load_pickle(proofs_path)
            with open(root_path) as f:
                root_hash = bytes.fromhex(f.read().strip())
            global_idx = np.where(labels == cid)[0]
            for local, g in enumerate(global_idx):
                st.chunk_meta[int(g)] = (proofs[local], root_hash)
            st.clusters[cid] = (faiss.read_index(index_path), global_idx)

        # Per-document centroids, computed from the stored MASKED embeddings.
        # The transform is orthogonal, so masked-query . masked-centroid equals
        # the raw similarity; no re-embedding needed (the old code embedded up to
        # 100 chunks per document one by one on the first query, and never
        # refreshed the cache after re-indexing).
        emb_path = os.path.join(CLUSTER_DIR, "transformed_embeddings.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            src_arr = np.array(st.sources)
            for src in set(st.sources):
                mean = emb[src_arr == src].mean(axis=0)
                st.source_centroids[src] = mean / (np.linalg.norm(mean) or 1.0)

        st.ready = bool(st.clusters)
        log(f"Loaded {len(st.chunks)} chunks, {len(st.clusters)} clusters, {len(st.source_centroids)} documents.")
        return st


def has_documents():
    return any(os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS for f in get_uploaded_files())


def get_uploaded_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f)))


index_lock = threading.RLock()     # guards re-indexing vs. queries
state = IndexState()


def reload_indexes():
    global state
    new_state = IndexState.load()
    with index_lock:
        state = new_state
    embed_masked_query.cache_clear()


# ------------------------------------------------------------------
# Conversations (persisted as JSON, guarded by a lock)
# ------------------------------------------------------------------
conv_lock = threading.Lock()


def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        try:
            with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
    return []


def save_conversations(convs):
    tmp = CONVERSATIONS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(convs, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CONVERSATIONS_FILE)   # atomic: no half-written file on crash


def create_conversation(title=""):
    return {"id": str(uuid.uuid4()), "title": title or "New Chat",
            "messages": [], "timestamp": datetime.now().isoformat()}


def get_or_create_conversation(conv_id):
    with conv_lock:
        convs = load_conversations()
        conv = next((c for c in convs if c["id"] == conv_id), None) if conv_id else None
        if conv is None:
            conv = create_conversation()
            convs.append(conv)
            save_conversations(convs)
        return conv


def append_to_conversation(conv_id, user_msg, assistant_msg):
    with conv_lock:
        convs = load_conversations()
        conv = next((c for c in convs if c["id"] == conv_id), None)
        if conv is None:
            conv = create_conversation()
            conv["id"] = conv_id
            convs.append(conv)
        if not conv["messages"]:
            q = user_msg["content"]
            conv["title"] = q[:40] + ("…" if len(q) > 40 else "")
        conv["messages"].extend([user_msg, assistant_msg])
        conv["timestamp"] = datetime.now().isoformat()
        save_conversations(convs)


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
_reranker = None
_reranker_failed = False


def get_reranker():
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"[RERANK] Unavailable: {e}")
            _reranker_failed = True
    return _reranker


@lru_cache(maxsize=512)
def embed_masked_query(query):
    """Client side: embed the query and mask it with the secret transform."""
    raw = embed_query(query)
    masked = apply_transform(raw, state.perm, state.signs)
    return normalize_vectors(masked)


def query_keywords(query, extra=None):
    words = re.findall(r'\b\w+\b', query.lower())
    kws = {w for w in words if w not in STOPWORDS and len(w) > 2}
    if extra:
        kws.update(k for k in extra if k and len(k) > 2)
    return sorted(kws)


def keyword_scores(st, keywords, top_k):
    """Lexical score in [0, 1]: fraction of keywords present, plus a phrase bonus."""
    if not keywords:
        return {}
    phrase = ' '.join(keywords)
    n = len(keywords)
    scores = []
    for idx, text in enumerate(st.chunks_lower):
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            s = hits / n
            if n > 1 and phrase in text:
                s += 0.5
            scores.append((s, idx))
    scores.sort(reverse=True)
    return {idx: s for s, idx in scores[:top_k]}


def is_factoid(query):
    synthesis = ('evolve', 'relationship', 'describe', 'explain', 'significance', 'compare',
                 'contrast', 'development', 'growth', 'transformation', 'summar', 'why')
    return not any(w in query.lower() for w in synthesis)


def retrieve_chunks(query):
    """Hybrid retrieval (masked dense + keyword) -> rerank -> Merkle verification.
    Returns (items, confidence) where items = [(chunk, source)]."""
    with index_lock:
        st = state
    if not st.ready:
        return [], 0.0

    masked_q = embed_masked_query(query)

    # ---- Cloud side: route to the closest clusters, search masked vectors ----
    sims = st.norm_centroids @ masked_q
    n_probe = min(NUM_CLUSTERS_TO_SELECT, len(sims))
    selected = np.argsort(-sims)[:n_probe]
    log(f"[CLUSTERS] probing {n_probe}/{len(sims)}: {selected.tolist()}")

    candidates = {}   # global idx -> {"dense": float, "kw": float}
    q = masked_q.reshape(1, -1)
    for cid in selected:
        if int(cid) not in st.clusters:
            continue
        index, global_idx = st.clusters[int(cid)]
        k = min(PER_CLUSTER_K, index.ntotal)
        dists, locs = index.search(q, k)
        for loc, d in zip(locs[0], dists[0]):
            if loc >= 0:
                candidates.setdefault(int(global_idx[loc]), {})["dense"] = float(d)

    # ---- Keyword channel ----
    extra = generate_keywords(query) if (USE_LLM_KEYWORDS and is_factoid(query) and len(query) < 40) else []
    if extra:
        log(f"[KEYWORDS] {extra}")
    for idx, s in keyword_scores(st, query_keywords(query, extra), KEYWORD_TOP_K).items():
        candidates.setdefault(idx, {})["kw"] = s

    # ---- Soft document boost ----
    best_source = None
    if len(st.source_centroids) > 1:
        best_source = max(st.source_centroids, key=lambda s: float(st.source_centroids[s] @ masked_q))
        log(f"[SOURCE BOOST] {best_source}")

    def base_score(idx, c):
        s = c.get("dense", 0.0) + 0.3 * c.get("kw", 0.0)
        if best_source and st.sources[idx] == best_source:
            s += SOURCE_BOOST
        return s

    # De-duplicate identical chunk texts, keep best-scoring copy
    ranked, seen = [], set()
    for idx, c in sorted(candidates.items(), key=lambda kv: base_score(*kv), reverse=True):
        if st.chunks[idx] in seen:
            continue
        seen.add(st.chunks[idx])
        ranked.append((idx, base_score(idx, c)))
    log(f"[CANDIDATES] {len(ranked)} unique")

    confidence = max((c.get("dense", 0.0) for c in candidates.values()), default=0.0)

    # ---- Rerank the head with a cross-encoder ----
    reranker = get_reranker() if USE_RERANK else None
    if reranker is not None and ranked:
        head = ranked[:RERANK_CANDIDATES]
        logits = reranker.predict([(query, st.chunks[i]) for i, _ in head], show_progress_bar=False)
        rescored = []
        for (idx, base), logit in zip(head, logits):
            prob = 1.0 / (1.0 + math.exp(-float(logit)))
            boost = SOURCE_BOOST if best_source and st.sources[idx] == best_source else 0.0
            rescored.append((idx, 0.8 * prob + 0.2 * base + boost, prob))
        rescored.sort(key=lambda x: x[1], reverse=True)
        confidence = rescored[0][2]
        ranked = [(i, s) for i, s, _ in rescored] + ranked[RERANK_CANDIDATES:]

    # ---- Client side: verify Merkle proofs, keep the first N that pass ----
    items = []
    for idx, _ in ranked:
        meta = st.chunk_meta.get(idx)
        if meta is None:
            continue
        ok = verify_merkle_proof(st.chunks[idx], meta[0], meta[1])
        log(f"[VERIFY] {st.sources[idx]} – {'PASSED' if ok else 'FAILED'}")
        if ok:
            items.append((st.chunks[idx], st.sources[idx]))
            if len(items) >= CONTEXT_CHUNKS:
                break
    return items, max(0.0, min(1.0, confidence)) * 100


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------
def build_messages(query, items, history, fallback=False):
    context = "\n\n".join(f"[{i}] (from {src}) {chunk}" for i, (chunk, src) in enumerate(items, 1))
    if fallback:
        system_msg = ("You are a helpful assistant. Answer the user's question based on the provided text. "
                      "Be concise but include the key reasons from the text, citing passages like [1]. "
                      "If the answer is not in the text, say 'I don't know' clearly.")
    else:
        system_msg = ("You are a precise assistant. Answer the user's question in 1-3 sentences using only "
                      "the numbered passages below, citing them like [1] or [2]. Then give up to two short "
                      "direct quotes from the passages as evidence. If the passages do not contain the answer, "
                      "say exactly 'I don't know'. If the question misspells a name, use the spelling in the text.")
    messages = [{"role": "system", "content": system_msg}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-HISTORY_TURNS:]]
    messages.append({"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {query}"})
    return messages


LLM_OPTIONS = {"num_ctx": LLM_NUM_CTX, "num_predict": LLM_MAX_TOKENS}


def is_greeting(text):
    t = re.sub(r'[^\w\s]', '', text.lower()).strip()
    # Whole-message match only: the old startswith() check treated
    # "history of..." / "highlights" as the greeting "hi".
    words = t.split()
    if t in GREETINGS:
        return True
    return (0 < len(words) <= 3 and words[0] in GREETINGS
            and not any(w in words for w in ("what", "how", "who", "why", "where", "when")))


def ndjson(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="Privacy-Preserving RAG", description="Verifiable RAG with a chat interface")


class QueryRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    model_config = ConfigDict(extra='ignore')


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int


# Bounded store of recent answers for feedback lookups
recent_answers: "OrderedDict[str, dict]" = OrderedDict()
MAX_RECENT = 500


def remember_answer(query_id, entry):
    recent_answers[query_id] = entry
    while len(recent_answers) > MAX_RECENT:
        recent_answers.popitem(last=False)


def canned_stream(conv_id, text, user_msg=None):
    yield ndjson({"type": "meta", "conversation_id": conv_id})
    yield ndjson({"type": "answer", "content": text})
    if user_msg:
        append_to_conversation(conv_id, {"role": "user", "content": user_msg},
                               {"role": "assistant", "content": text})
    yield ndjson({"type": "done", "confidence": None, "query_id": "", "conversation_id": conv_id})


# NOTE: endpoints that do blocking work (embedding, Ollama, indexing) are plain
# `def`, so FastAPI runs them in a worker thread. The old `async def` versions
# blocked the whole server while the LLM was generating or indexing ran.
@app.post("/chat")
def chat_endpoint(req: QueryRequest):
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty message")
    log("\n" + "=" * 60 + f"\n[QUERY] {query}\n" + "=" * 60)

    conv = get_or_create_conversation(req.conversation_id)
    conv_id = conv["id"]
    history = conv["messages"]

    if is_greeting(query):
        return StreamingResponse(canned_stream(conv_id, "Hello! Ask me anything about your uploaded documents.", query),
                                 media_type="application/x-ndjson")
    if not state.ready:
        return StreamingResponse(canned_stream(conv_id, "No documents are indexed yet. Upload a document from the sidebar first."),
                                 media_type="application/x-ndjson")

    def event_stream():
        yield ndjson({"type": "meta", "conversation_id": conv_id})
        yield ndjson({"type": "status", "content": "Searching documents…"})
        try:
            items, confidence = retrieve_chunks(query)
        except Exception as e:
            yield ndjson({"type": "error", "content": f"Retrieval failed: {e}"})
            return
        if not items:
            yield ndjson({"type": "answer", "content": "I couldn't find anything about that in the uploaded documents. "
                                                       "Try rephrasing, or upload a document that covers it."})
            yield ndjson({"type": "done", "confidence": 0, "query_id": "", "conversation_id": conv_id})
            return

        citations = [{"id": i, "source": src, "chunk": chunk[:600] + ("…" if len(chunk) > 600 else "")}
                     for i, (chunk, src) in enumerate(items, 1)]
        yield ndjson({"type": "citations", "citations": citations})
        yield ndjson({"type": "status", "content": "Writing answer…"})

        answer = ""
        try:
            for part in ollama.chat(model=LLM_MODEL, messages=build_messages(query, items, history),
                                    stream=True, options=LLM_OPTIONS):
                token = part.get('message', {}).get('content', '')
                if token:
                    answer += token
                    yield ndjson({"type": "answer", "content": token})
        except Exception as e:
            yield ndjson({"type": "error", "content": f"LLM error: {e}. Is Ollama running with '{LLM_MODEL}' pulled?"})
            return

        if len(answer.split()) < 5 or "i don't know" in answer.lower():
            log("[FALLBACK] Initial answer insufficient, retrying with a looser prompt")
            yield ndjson({"type": "status", "content": "Trying a broader answer…"})
            try:
                resp = ollama.chat(model=LLM_MODEL, messages=build_messages(query, items, history, fallback=True),
                                   options=LLM_OPTIONS)
                fallback = resp['message']['content'].strip()
                if len(fallback.split()) > 3 and "i don't know" not in fallback.lower():
                    answer = fallback
                    # Replace instead of appending (the old code streamed both answers back to back).
                    yield ndjson({"type": "replace", "content": answer})
            except Exception as e:
                log(f"[FALLBACK] Failed: {e}")

        query_id = str(uuid.uuid4())
        remember_answer(query_id, {"query": query, "chunks": [c for c, _ in items], "answer": answer,
                                   "confidence": confidence})
        append_to_conversation(conv_id, {"role": "user", "content": query},
                               {"role": "assistant", "content": answer, "citations": citations,
                                "query_id": query_id})
        log_query(query, [c for c, _ in items], answer, confidence=confidence, query_id=query_id)
        yield ndjson({"type": "done", "confidence": round(confidence, 1), "query_id": query_id,
                      "conversation_id": conv_id})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ---------- Conversations ----------
@app.get("/conversations")
def get_conversations():
    convs = sorted(load_conversations(), key=lambda c: c.get("timestamp", ""), reverse=True)
    return [{
        "id": c["id"],
        "title": c["title"],
        "timestamp": c.get("timestamp", ""),
        "last_message": (c["messages"][-1]["content"][:60] if c["messages"] else ""),
    } for c in convs]


@app.get("/conversation/{conv_id}")
def get_conversation(conv_id: str):
    conv = next((c for c in load_conversations() if c["id"] == conv_id), None)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.post("/new_conversation")
def new_conversation():
    conv = get_or_create_conversation(None)
    return {"id": conv["id"], "title": conv["title"]}


@app.delete("/delete_conversation/{conv_id}")
def delete_conversation(conv_id: str):
    with conv_lock:
        save_conversations([c for c in load_conversations() if c["id"] != conv_id])
    return {"status": "deleted"}


# ---------- Documents ----------
def safe_filename(name):
    """Strip directories so uploads/deletes cannot escape DATA_DIR (path traversal)."""
    name = os.path.basename((name or "").replace("\\", "/")).strip()
    if not name or name in (".", "..") or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid file name")
    return name


def reindex():
    with index_lock:
        index_main()
        reload_indexes()


@app.post("/upload")
def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    saved, skipped = [], []
    for file in files:
        name = safe_filename(file.filename)
        if os.path.splitext(name)[1].lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(name)
            continue
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            skipped.append(name)
            continue
        with open(os.path.join(DATA_DIR, name), "wb") as f:
            f.write(data)
        saved.append(name)
    if not saved:
        raise HTTPException(status_code=400, detail=f"No supported files (skipped: {', '.join(skipped)})")
    try:
        reindex()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    msg = f"Indexed {len(saved)} file(s)."
    if skipped:
        msg += f" Skipped unsupported/too large: {', '.join(skipped)}."
    return {"status": "success", "message": msg, "files": get_uploaded_files()}


@app.delete("/delete_file/{filename}")
def delete_file(filename: str):
    path = os.path.join(DATA_DIR, safe_filename(filename))
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    os.remove(path)
    try:
        reindex()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {e}")
    return {"status": "success", "message": f"Deleted {filename}", "files": get_uploaded_files()}


@app.get("/files")
def list_files():
    return {"files": get_uploaded_files()}


@app.get("/status")
def status():
    st = state
    return {"ready": st.ready, "files": len(get_uploaded_files()), "chunks": len(st.chunks),
            "clusters": len(st.clusters), "llm": LLM_MODEL, "embedding_model": EMBEDDING_MODEL}


# ---------- Feedback ----------
@app.post("/feedback")
def feedback_endpoint(fb: FeedbackRequest):
    entry = recent_answers.get(fb.query_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Query ID not found")
    new_file = not os.path.exists(FEEDBACK_LOG) or os.path.getsize(FEEDBACK_LOG) == 0
    with open(FEEDBACK_LOG, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "query_id", "query", "rating", "chunks", "answer", "confidence"])
        writer.writerow([datetime.now().isoformat(), fb.query_id, entry["query"], fb.rating,
                         "|".join(entry["chunks"]), entry["answer"], entry["confidence"]])
    return {"status": "feedback recorded"}


# ---------- UI ----------
@app.get("/")
def serve_ui():
    return FileResponse(UI_FILE, media_type="text/html")


print("Loading indexes...")
reload_indexes()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
