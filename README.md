# Privacy-Preserving RAG

A local Retrieval-Augmented Generation (RAG) demo that indexes documents, retrieves passages with FAISS, verifies retrieved chunks using Merkle proofs, and drafts answers with a locally running Ollama model.

> **Privacy note:** This is a research and demonstration project, not a security certification. Permutation and sign-flip embedding masking is not encryption, and document chunks and metadata are stored locally in index files. Review the implementation and threat model before using sensitive documents or exposing the web interface to a network.

## Overview

This project includes a browser-based chat demo built with FastAPI and a command-line pipeline. Documents are extracted, chunked, embedded, clustered, masked, and indexed. Queries are masked, matched against relevant clusters, reranked, verified, and passed to a local language model that generates answers from the retrieved context.

## Features

- Local document ingestion for PDF, DOCX, TXT, Markdown, and common image formats. Image OCR requires Tesseract.
- Sentence-transformer embeddings, clustering, and FAISS similarity search.
- Permutation and sign-flip embedding masking with Merkle-proof verification of retrieved chunks.
- Local answer generation and query rewriting through Ollama.
- Browser-based interface with document uploads, conversation history, and feedback logging.
- Command-line pipeline for indexing and querying documents.
- Local index storage to support retrieval without relying on a hosted vector database.

## Tech Stack

- **Language:** Python 3
- **Web framework:** FastAPI
- **ASGI server:** Uvicorn
- **Retrieval and machine learning:** Sentence Transformers, PyTorch, FAISS CPU, scikit-learn, NumPy
- **LLM runtime:** Ollama, using `llama3.1:8b` by default
- **Document processing:** PyMuPDF, python-docx, Pillow, pytesseract
- **Supporting libraries:** Pydantic, python-multipart, tqdm, and other dependencies listed in `requirements.txt`

## System Architecture

1. **Document ingestion:** Read supported files and extract their text.
2. **Chunking and embedding:** Split extracted text into chunks and generate vector embeddings.
3. **Clustering and masking:** Organize embeddings into clusters and apply the project's permutation and sign-flip masking transformation.
4. **Indexing:** Store retrieval data in a FAISS index and maintain verification data for retrieved chunks.
5. **Query processing:** Process and mask a user query, then retrieve and rerank relevant chunks.
6. **Verification:** Verify retrieved chunks using the project's Merkle-proof mechanism.
7. **Answer generation:** Send retrieved context to a locally running Ollama model to draft an answer.

The masking and verification mechanisms are project-specific techniques; they should not be interpreted as guarantees of confidentiality, integrity, or resistance to attacks.

## Project Structure

```text
privacy-rag/
├── demo.py                  # FastAPI web demo
├── requirements.txt         # Python dependencies
├── ui/
│   └── index.html           # Browser interface
├── data/
│   └── raw/                 # Source documents for CLI indexing
└── src/
    ├── config.py            # Paths, model names, and tuning constants
    ├── pipeline.py          # Command-line orchestration
    ├── api/                 # API package structure
    ├── client/              # Query preprocessing and masking
    ├── cloud/               # FAISS retrieval utilities
    ├── indexing/             # Document indexing and clustering
    ├── llm/                 # LLM package structure
    ├── utils/               # Extraction, embeddings, logging, and helpers
    └── verification/        # Merkle verification and masking validation
```

The `api/` and `llm/` directories currently contain package initialization files. Generated indexes are written to `indexes/` and should not be committed. The exact generated files depend on the pipeline and configuration.

## Prerequisites

- Python 3.10 or newer.
- [Ollama](https://ollama.com/) installed and running locally.
- Sufficient RAM and disk space for the embedding model (`BAAI/bge-large-en-v1.5`) and the selected Ollama model.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract), if indexing images.
- Operating-system build tools if a Python dependency needs compilation.

## Installation and Setup

Run these commands from the project root.

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start Ollama and download the model

Ensure Ollama is installed and running, then execute:

```bash
ollama pull llama3.1:8b
```

The first indexing run may download the embedding model, which can take some time and disk space.

## Environment Variables

The project does not use a `.env` file by default. The following optional variables are read directly from the process environment.

| Variable | Purpose | Default |
|---|---|---|
| `RAG_LLM_MODEL` | Ollama model used for answer generation and query rewriting | `llama3.1:8b` |
| `RAG_VERBOSE` | Controls verbose logging in the web demo | `1` |

Example on macOS/Linux:

```bash
RAG_LLM_MODEL=mistral python demo.py
```

Example in Windows PowerShell:

```powershell
$env:RAG_LLM_MODEL="mistral"
python demo.py
```

The embedding model and indexing parameters are configured in `src/config.py`.

## Usage

### Web Demo

Start the web application:

```bash
python demo.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser, upload documents through the interface, and ask questions about their content.

**Security warning:** The current demo binds Uvicorn to `0.0.0.0`, which can make it reachable from other machines on the network. Its upload and delete endpoints do not have authentication. Run it only in a trusted environment, or change the host to `127.0.0.1` in `demo.py` for local-only access.

### Command-Line Pipeline

Place supported source documents in `data/raw/`, then run:

```bash
# Index documents if needed, then prompt for a query
python -m src.pipeline

# Force the index to be rebuilt
python -m src.pipeline --reindex
```

### Optional Utilities

The repository also provides utility modules that can be run as Python modules:

```bash
python -m src.indexing.build_index
python -m src.verification.masking_validation
python -m src.utils.visualization
```

These utilities may require an existing index. Check their module implementations and configuration before running them.

### Testing

No automated test suite is included in the current project. A basic Python syntax check can be run with:

```bash
python -m compileall src demo.py
```

This checks syntax but does not verify retrieval quality, masking effectiveness, or security.

## Deployment

This project depends on local models, FAISS indexes stored on disk, and a running Ollama service. It cannot run as a static website on GitHub Pages.

For a hosted deployment, use a Python-capable server with persistent storage, sufficient memory for the models, and access to an Ollama instance. The FastAPI application can be served through Uvicorn or another ASGI server behind a properly configured reverse proxy.

Before exposing the application beyond localhost:

- Add authentication and authorization.
- Apply file-upload size and request-rate limits.
- Configure HTTPS and network restrictions.
- Restrict access to uploaded documents, indexes, logs, and conversation history.
- Review the threat model and validate the masking and verification mechanisms independently.

The demo is not a hardened multi-user service. Do not commit private source documents, generated indexes, query logs, conversation history, credentials, or other sensitive information.

## Future Improvements

- Add automated tests for document ingestion, retrieval, verification, and answer generation.
- Add authentication and authorization for web endpoints.
- Improve upload validation, error handling, and resource limits.
- Evaluate retrieval quality and latency on a documented test dataset.
- Document the threat model and independently evaluate the masking mechanism.
- Add reproducible benchmarks and structured logging with sensitive-data controls.
- Provide deployment configuration and operational guidance.

## Contributing

1. Create a feature branch for your changes.
2. Keep changes focused and document behavior changes.
3. Run `python -m compileall src demo.py` and any relevant checks before opening a pull request.
4. Never commit private documents, generated indexes, logs, or secrets.
5. Describe known limitations and security implications when modifying privacy-related features.

## License

A license has not yet been added. The license file will be added when licensing is finalized for the project repositories. Until then, do not assume that the code is available for unrestricted reuse.
