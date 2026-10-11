import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
CLUSTER_DIR = os.path.join(INDEX_DIR, "clusters")
FAISS_DIR = os.path.join(INDEX_DIR, "faiss")
MERKLE_DIR = os.path.join(INDEX_DIR, "merkles")
TRANSFORM_DIR = os.path.join(INDEX_DIR, "transform")
# Per-file cache of MASKED embeddings so re-indexing only embeds new/changed files.
EMBED_CACHE_DIR = os.path.join(INDEX_DIR, "embed_cache")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
N_CLUSTERS_MAX = 64
TOP_K = 50
BATCH_SIZE = 64
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ***** BEST MODEL *****
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "llama3.1:8b")   # or "mistral"

# Ollama defaults to a 2k-4k token window, which silently truncated the
# retrieved context (15 x 512-word chunks is ~10k tokens). Keep these in sync.
LLM_NUM_CTX = 8192
LLM_MAX_TOKENS = 1024
CONTEXT_CHUNKS = 8          # chunks handed to the LLM per answer

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

USE_LLM_REWRITE = True
# Only probe the closest clusters instead of every cluster (the point of clustering).
NUM_CLUSTERS_TO_SELECT = 8

ADAPTIVE_FAST_THRESHOLD = 30
ADAPTIVE_BALANCED_THRESHOLD = 60
LATENCY_MODE = "adaptive"

FEEDBACK_LOG = os.path.join(INDEX_DIR, "feedback_log.csv")

for d in [DATA_DIR, CLUSTER_DIR, FAISS_DIR, MERKLE_DIR, TRANSFORM_DIR, EMBED_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
