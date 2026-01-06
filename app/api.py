from fastapi import APIRouter, UploadFile, File, HTTPException
from app.chunking import semantic_chunk
from app.db import get_connection
from app.embedding import embed_text
import pickle


router = APIRouter()

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files allowed")

    content = await file.read()
    text = content.decode("utf-8")

    chunks = semantic_chunk(text)

    conn = get_connection()
    cursor = conn.cursor()

    for idx, chunk in enumerate(chunks):
        
        embedding = embed_text(chunk)
        embedding_blob = pickle.dumps(embedding)

        cursor.execute(
                "INSERT INTO documents (document_name, chunk_id, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                (file.filename, idx, chunk, embedding_blob)
        )

    conn.commit()
    conn.close()

    return {
        "document": file.filename,
        "chunks_stored": len(chunks),
        "status": "success"
    }

@router.get("/health")
def health():
    try:
        conn = get_connection()
        conn.close()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "error", "details": str(e)}
