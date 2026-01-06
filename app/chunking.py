import re

MAX_CHARS = 500

def semantic_chunk(text: str):
    paragraphs = text.split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= MAX_CHARS:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?]) +', para)
            current = ""

            for sentence in sentences:
                if len(current) + len(sentence) <= MAX_CHARS:
                    current += " " + sentence
                else:
                    chunks.append(current.strip())
                    current = sentence

            if current:
                chunks.append(current.strip())

    return chunks
