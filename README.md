# Privacy-Preserving RAG

A local retrieval-augmented generation (RAG) demo that indexes documents, retrieves passages with FAISS, verifies retrieved chunks with Merkle proofs, and drafts answers with a locally running Ollama model.

> **Privacy note:** This is a research and demo project, not a security certification. Embedding permutation and sign-flip masking is not encryption, and document chunks and metadata are stored locally in index files. Review the implementation and threat model before using sensitive documents or exposing the web UI to a network.

## Overview

The project includes a browser-based chat demo (FastAPI) and a command-line pipeline. Documents are extracted, chunked, embedded, clustered, masked, and indexed. Queries are masked, matched against the closest clusters, reranked, verified, and passed to a local LLM that answers from the retrieved context.

## Features

- Local document ingestion for PDF, DOCX, TXT, Markdown, and common image formats (OCR requires Tesseract).
- Sentence-transformer embeddings, clustering, and FAISS similarity search.
- Permutation and sign-flip embedding transform, with Merkle-proof verification of retrieved chunks.
- Local answer generation and query rewriting through Ollama.
- Web interface with document upload, conversation history, and feedback logging.
- Command-line pipeline for indexing and querying.

## Tech Stack

- **Language:** Python 3
- **Web:** FastAPI, Uvicorn
- **Retrieval and ML:** sentence-transformers, PyTorch, FAISS (CPU), scikit-learn, NumPy
- **LLM runtime:** Ollama (default model `llama3.1:8b`)
- **Document processing:** PyMuPDF, python-docx, Pillow, pytesseract

## Project Structure

```text
privacy-rag/
├── demo.py                 # FastAPI web demo (serves ui/index.html on port 8000)
├── requirements.txt        # Python dependencies
├── ui/
│   └── index.html          # Browser interface
├── data/
│   └── raw/                # Put source documents here for the CLI pipeline (contents not committed)
└── src/
    ├── config.py           # Paths, model names, and tuning constants
    ├── pipeline.py         # CLI orchestration
    ├── indexing/           # Document indexing and clustering
    ├── client/             # Query preprocessing and masking
    ├── cloud/              # FAISS retrieval
    ├── verification/       # Merkle verification, answer generation, masking validation
    └── utils/              # Extraction, embeddings, helpers, logging, visualization
```

Generated indexes are written to `indexes/` (not committed).

## Prerequisites

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- Enough RAM and disk space for the embedding model (`BAAI/bge-large-en-v1.5`) and the chosen Ollama model.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), only if you index images.
- Build tools for your OS, if a Python dependency has to be compiled.

## Installation

Run these commands from the project root.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.1:8b
```

The first indexing run downloads the embedding model, which can take some time.

## Environment Variables

No `.env` file is used; the application does not load one. Both variables below are optional and are read directly from the process environment.

| Variable | Purpose | Default |
| --- | --- | --- |
| `RAG_LLM_MODEL` | Ollama model used for answers and query rewriting | `llama3.1:8b` |
| `RAG_VERBOSE` | Set to `1` for verbose web-demo logging | `1` |

Example: `RAG_LLM_MODEL=mistral python demo.py` (Windows PowerShell: `$env:RAG_LLM_MODEL="mistral"; python demo.py`).

The embedding model and indexing parameters are set in `src/config.py`.

## Usage

### Web demo

```bash
python demo.py
```

Open <http://127.0.0.1:8000>, upload documents through the interface, and ask questions about them.

Note: `demo.py` starts Uvicorn bound to `0.0.0.0`, so the server is reachable from other machines on your network, and its upload and delete endpoints have no authentication. Run it only on a trusted network, or change the host to `127.0.0.1` in the last line of `demo.py`.

### Command-line pipeline

Place documents in `data/raw/`, then run:

```bash
python -m src.pipeline            # Indexes if no index exists, then prompts for a query
python -m src.pipeline --reindex  # Force a rebuild of the index
```

### Optional utilities

```bash
python -m src.indexing.build_index
python -m src.verification.masking_validation
python -m src.utils.visualization
```

The validation and visualization utilities expect an index to exist first.

Tests: no automated test suite is included.

## Deployment

This project depends on local models, FAISS indexes on disk, and a running Ollama service. GitHub Pages cannot run it.

To host it, use a Python-capable server with persistent storage, enough memory for the models, and access to an Ollama instance, then run `python demo.py` (or serve the `demo:app` ASGI app with Uvicorn) behind a reverse proxy.

Before exposing it beyond localhost, add authentication and authorization, request and upload limits, HTTPS, and network restrictions. The demo is not a hardened multi-user service. Do not commit private source documents, generated indexes, query logs, or conversation history.

## Contributing

1. Create a feature branch.
2. Keep changes focused and document behavior changes.
3. Run `python -m compileall src demo.py` and any other checks you have before opening a pull request.
4. Never commit private documents, generated indexes, logs, or secrets.

## License

No `LICENSE` file is included. Add one that states the terms you intend before publishing or distributing this project.
