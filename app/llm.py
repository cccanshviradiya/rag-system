import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found")


# Create client ONCE (important for performance)
client = genai.Client(api_key=API_KEY)


def generate_answer(question: str, context_chunks: list) -> str:
    if not context_chunks:
        return "I don’t know based on the provided context."

    context_text = "\n\n".join(
        f"- {chunk['text']}" for chunk in context_chunks
    )

    prompt = f"""
You are answering a question using retrieved context from documents.

Rules:
- Use the context as the primary source.
- If the answer is partially present, infer carefully.
- If the answer is truly not present, say:
  "I don’t know based on the provided context."

Context:
{context_text}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text.strip()



