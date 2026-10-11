# src/utils/logger.py
import csv
import os
from datetime import datetime
from src.config import INDEX_DIR

LOG_FILE = os.path.join(INDEX_DIR, "query_log.csv")

def log_query(query, chunks, answer, confidence=0.0, query_id=""):
    """Log query with confidence and query_id."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "query_id", "query", "chunks_retrieved", "answer", "confidence"])
        chunks_str = " | ".join(chunks)
        writer.writerow([datetime.now().isoformat(), query_id, query, chunks_str, answer, confidence])