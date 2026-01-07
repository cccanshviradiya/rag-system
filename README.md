# Retrieval-Augmented Generation (RAG) System – From Scratch

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system from scratch using Python and FastAPI.  
The system ingests plain text documents, performs **semantic chunking**, generates embeddings using **Hugging Face models**, retrieves relevant context using **manual cosine similarity**, and answers user questions using a **Gemini LLM**, strictly grounded in retrieved evidence.

The solution strictly follows the given constraints:
- No LangChain / LlamaIndex
- No vector databases
- No answers without evidence
- Confidence must be computed (not hardcoded)

---

## System Architecture
```bash

Text File (.txt)
   ↓
Semantic Chunking
   ↓
Embedding (Hugging Face)
   ↓
SQLite Storage
   ↓
Semantic Retrieval (Cosine Similarity)
   ↓
LLM Prompt (Context-Only)
   ↓
Answer + Evidence + Confidence


```
---

## API Endpoints

| Method | Endpoint  | Description |
|------|----------|------------|
| POST | `/ingest` | Upload and index a `.txt` document |
| POST | `/ask`    | Ask a question using RAG |
| GET  | `/health` | Health check and diagnostics |

---

## API Response Format

```json
{
  "question": "string",
  "answer": "string",
  "confidence": 0.0,
  "evidence": [
    {
      "document": "doc.txt",
      "chunk_id": 2,
      "text": "relevant excerpt"
    }
  ]
}

```
---

## Chunking Strategy

Semantic chunking is performed using:

- Paragraph-level splitting (`\n\n`)
- Sentence-level splitting for oversized paragraphs
- A maximum chunk size of **500 characters**

This approach preserves semantic meaning better than line-based or fixed-length chunking.

---

## Embedding Choice

- **Model:** `intfloat/e5-small-v2`
- **Source:** Hugging Face

### Reasoning
- Lightweight and fast
- Produces high-quality sentence embeddings
- Free and locally executable
- Industry-standard for semantic search

Embeddings are generated once per chunk and stored locally.

---

## Database Choice

- **Database:** SQLite

### Reasoning
- Lightweight and file-based
- No external dependencies
- Fully compliant with the **“no vector DB”** constraint
- Sufficient for embedding storage at small–medium scale

### Stored Fields
- `document_name`
- `chunk_id`
- `chunk_text`
- `embedding` (BLOB)

---

## Retrieval Logic

- The user question is embedded using the same embedding model
- All stored chunk embeddings are loaded from SQLite
- Cosine similarity is computed manually using NumPy
- Chunks are ranked by similarity
- Top-k (default = 5) chunks are selected

No approximate search, indexing library, or vector database is used.

---

## LLM Integration

- **Model:** Gemini (via official REST API)

### Why REST Instead of SDK
- Avoids SDK version instability
- Provides full control over API version
- Production-safe and deterministic behavior

---

## Hallucination Prevention

Hallucination is prevented using strict prompt constraints:

- The LLM is instructed to answer **only** from retrieved context
- If the answer is not present, it must respond exactly:
- No external knowledge is allowed

This ensures all answers are fully traceable to retrieved evidence.

---

## Confidence Computation

Confidence is **computed**, not hardcoded.

### Method
- Compute cosine similarity for each retrieved chunk
- Calculate the average similarity
- Normalize for interpretability:
  
 ```bash
  confidence = min(1.0, average_similarity × 2.5)
```

---

### Interpretation
- Reflects semantic relevance, not factual certainty
- Normalization improves human interpretability
- Preserves ranking and mathematical integrity

---

## Health Check

The `/health` endpoint validates:

- Database connectivity
- Embedding model availability
- LLM configuration

This helps diagnose failures instantly.

---

## Limitations

- SQLite does not scale for very large datasets
- No chunk overlap is implemented
- No multilingual support
- Embedding-based similarity may assign low scores to short factual questions

These trade-offs were intentionally accepted to meet the given constraints.

---

## How to Run

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload






