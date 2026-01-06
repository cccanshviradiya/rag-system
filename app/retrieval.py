import pickle

from app.db import get_connection
from app.embedding import embed_text, cosine_similarity


def retrieve_top_k_chunks(question: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve top-k semantically relevant chunks for a given question.
    """

    question_embedding = embed_text(question)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT document_name, chunk_id, chunk_text, embedding
        FROM documents
        """
    )

    rows = cursor.fetchall()
    conn.close()

    results = []

    for document, chunk_id, text, emb_blob in rows:
        chunk_embedding = pickle.loads(emb_blob)

        similarity = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        results.append({
            "document": document,
            "chunk_id": chunk_id,
            "text": text,
            "similarity": similarity
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]




