# Vector DB 🦀

A high-performance semantic search engine built with Rust + BERT embeddings.

## Stack
- **Rust** + **axum** — async web server
- **candle** — BERT embeddings (sentence-transformers/all-MiniLM-L6-v2)
- **anyhow** — error handling
- **uuid** — unique document IDs

## Features
- ✅ Insert documents with automatic embedding generation
- ✅ Semantic search with cosine similarity
- ✅ List all documents
- ✅ Delete documents by ID
- ✅ Health check

## Setup

### 1. Download model
```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='model')
"
```

### 2. Run
```bash
cargo run --release
```

## API

### `POST /insert`
Add a document to the database.

**Request:**
```json
{"text": "Rust is a systems programming language"}
```

**Response:**
```json
{
  "id": "ce56c023-b65a-4f14-a022-510feec46892",
  "message": "Document inserted successfully"
}
```

### `POST /search`
Semantic search over all documents.

**Request:**
```json
{"query": "machine learning and AI", "top_k": 3}
```

**Response:**
```json
{
  "results": [
    {"id": "abc123", "text": "Python is great for machine learning", "score": 0.91},
    {"id": "def456", "text": "Neural networks are inspired by the brain", "score": 0.79}
  ],
  "total_docs": 4
}
```

### `GET /documents`
List all documents in the database.

**Response:**
```json
{
  "documents": [
    {"id": "abc123", "text": "Rust is a systems programming language"},
    {"id": "def456", "text": "Python is great for machine learning"}
  ],
  "total_docs": 2
}
```

### `DELETE /delete/:id`
Delete a document by ID.

**Response:**
```json
{
  "id": "abc123",
  "message": "Document deleted successfully"
}
```

### `GET /health`
Health check.

**Response:**
```json
{"status": "ok", "version": "0.1.0", "total_docs": 4}
```

## Error Handling
All errors return a JSON response:
```json
{"error": "Document with id 'abc123' not found"}
```
