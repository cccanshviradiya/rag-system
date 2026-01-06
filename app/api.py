from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pickle

from app.chunking import semantic_chunk
from app.db import get_connection
from app.embedding import embed_text
from app.retrieval import retrieve_top_k_chunks
from app.llm import generate_answer
from app.confidence import compute_confidence

router = APIRouter()


@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a plain text document, chunk it, embed it,
    and store results in the database.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files allowed")

    text = (await file.read()).decode("utf-8")
    chunks = semantic_chunk(text)

    conn = get_connection()
    cursor = conn.cursor()

    for idx, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        embedding_blob = pickle.dumps(embedding)

        cursor.execute(
            """
            INSERT INTO documents (document_name, chunk_id, chunk_text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (file.filename, idx, chunk, embedding_blob),
        )

    conn.commit()
    conn.close()

    return {
        "document": file.filename,
        "chunks_stored": len(chunks),
        "status": "success",
    }


class QuestionRequest(BaseModel):
    question: str


@router.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Answer a question using retrieval-augmented generation.
    """
    question = request.question
    top_chunks = retrieve_top_k_chunks(question, top_k=3)

    if not top_chunks:
        return {
            "question": question,
            "answer": "I don’t know based on the provided context.",
            "confidence": 0.0,
            "evidence": [],
        }

    answer = generate_answer(question, top_chunks)
    confidence = compute_confidence(top_chunks)

    evidence = [
        {
            "document": chunk["document"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in top_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
    }


@router.get("/health")
def health():
    """
    Health check endpoint.
    """
    try:
        conn = get_connection()
        conn.close()
        return {
            "status": "ok",
            "db": "connected",
            "embedding_model": "loaded",
            "llm": "gemini",
        }
    except Exception as e:
        return {
            "status": "error",
            "details": str(e),
        }
