def compute_confidence(retrieved_chunks: list) -> float:
    """
    Compute confidence based on normalized average cosine similarity
    of retrieved chunks.
    """
    if not retrieved_chunks:
        return 0.0

    avg_similarity = sum(
        chunk["similarity"] for chunk in retrieved_chunks
    ) / len(retrieved_chunks)

    # Normalize similarity for human-friendly confidence score
    confidence = min(1.0, avg_similarity * 2.5)

    return round(confidence, 2)
