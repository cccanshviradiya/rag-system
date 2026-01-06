import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")


def generate_answer(question: str, context_chunks: list) -> str:

    if not context_chunks:
        return "I don’t know based on the provided context."

    client = genai.Client(api_key=API_KEY)

    context_text = "\n\n".join(
        f"- {chunk['text']}" for chunk in context_chunks
    )

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say exactly:
"I don’t know based on the provided context."

Context:
{context_text}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


