from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pickle
from typing import List

from app.chunking import semantic_chunk
from app.db import get_connection
from app.embedding import embed_text
from app.retrieval import retrieve_top_k_chunks
from app.llm import generate_answer
from app.confidence import compute_confidence

router = APIRouter()

@router.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(...)):

    conn = get_connection()
    cursor = conn.cursor()

    total_chunks = 0
    ingested_files = []

    for file in files:
        if not file.filename.endswith(".txt"):
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="Only .txt files allowed"
            )

        text = (await file.read()).decode("utf-8")
        chunks = semantic_chunk(text)

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

        total_chunks += len(chunks)
        ingested_files.append(file.filename)

    conn.commit()
    conn.close()

    return {
        "documents": ingested_files,
        "total_chunks_stored": total_chunks,
        "status": "success",
    }



class QuestionRequest(BaseModel):
    question: str
    top_k: int | None = None 


@router.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question
    top_k = request.top_k or 5
    top_k = max(1, min(top_k, 20))
    top_chunks = retrieve_top_k_chunks(question, top_k=top_k)

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
