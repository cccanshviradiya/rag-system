import numpy as np
from sentence_transformers import SentenceTransformer

from sentence_transformers import SentenceTransformer
import numpy as np

# Load once
model = SentenceTransformer("intfloat/e5-small-v2")


def embed_text(text: str, is_query: bool = False) -> np.ndarray:
    """
    E5 requires prefixes:
    - 'query:' for questions
    - 'passage:' for document chunks
    """
    prefix = "query: " if is_query else "passage: "
    embedding = model.encode(prefix + text, normalize_embeddings=True)
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    a = np.asarray(a)
    b = np.asarray(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)
