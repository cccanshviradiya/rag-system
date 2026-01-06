# import pickle
# from app.db import get_connection
# from app.embedding import cosine_similarity

# def retrieve_top_k_chunks(question_embedding,top_k=int 3):
#     conn = get_connection()
#     cursor = conn.cursor()

#     cursor.execute("""
#         SELECT document_name, chunk_id, chunk_text, embedding
#         FROM documents
#     """)

#     rows = cursor.fetchall()
#     conn.close()

#     scored_chunks = []

#     for doc, chunk_id, text, emb_blob in rows:
#         embedding = pickle.loads(emb_blob)
#         score = cosine_similarity(question_embedding, embedding)

#         scored_chunks.append({
#             "document": doc,
#             "chunk_id": chunk_id,
#             "text": text,
#             "score": score
#         })

#     scored_chunks.sort(key=lambda x: x["score"], reverse=True)

#     return scored_chunks[:top_k]
#     # return scored_chunks[:k]

import numpy as np
import pickle

from app.db import get_connection
from app.embedding import embed_text


def cosine_similarity(vec1, vec2):
    """
    Manual cosine similarity using NumPy
    """
    numerator = np.dot(vec1, vec2)
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def retrieve_top_k_chunks(question: str, top_k: int = 3):
    """
    Semantic retrieval:
    - Embed the question
    - Compare with stored chunk embeddings
    """

    # 🔑 THIS WAS MISSING CONCEPTUALLY BEFORE
    question_embedding = embed_text(question)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT document_name, chunk_id, chunk_text, embedding
        FROM documents
    """)
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

