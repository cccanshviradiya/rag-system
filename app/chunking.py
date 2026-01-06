import re

MAX_CHARS = 500


def semantic_chunk(text: str) -> list[str]:
    """
    Split text into semantic chunks using paragraphs and sentence boundaries.
    """
    chunks = []

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue

        # If paragraph is small enough, keep as is
        if len(para) <= MAX_CHARS:
            chunks.append(para)
            continue

        # Otherwise, split into sentence-based chunks
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= MAX_CHARS:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

    return chunks
