import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once at startup
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """
    Convert input text into an embedding vector.
    """
    return model.encode(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)
