import numpy as np
from sentence_transformers import SentenceTransformer

# Load model ONCE
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# def embed_text(text: str):
#     """Convert text into embedding vector"""
#     embedding = model.encode(text)
#     return embedding
def embed_text(text: str) -> np.ndarray:
    """
    Convert text into an embedding vector
    """
    return model.encode(text)


def cosine_similarity(a, b):
    """Return cosine similarity between two vectors as a float."""
    a = np.array(a)
    b = np.array(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
