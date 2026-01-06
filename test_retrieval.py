"""
Manual test file for retrieval logic.
Run this AFTER:
1. /ingest has been used at least once
2. Database contains embeddings
"""

from app.retrieval import retrieve_top_k_chunks

def main():
    question = "What is rag?"

    print("\nQUESTION:")
    print(question)

    print("\nRetrieving top chunks...\n")

    results = retrieve_top_k_chunks(question, top_k=3)

    for i, item in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Document   : {item['document']}")
        print(f"Chunk ID   : {item['chunk_id']}")
        print(f"Similarity : {item['similarity']:.4f}")
        print("Text:")
        print(item['text'])
        print()

if __name__ == "__main__":
    main()


"""
Semantic Retrieval Test
This file verifies that retrieval is based on meaning (embeddings),
not keyword matching.
"""

# from app.retrieval import retrieve_top_k_chunks

# def main():
#     # Use a question that does NOT directly match document words
#     question = "Explain what AI is"

#     print("=" * 60)
#     print("SEMANTIC SEARCH TEST")
#     print("=" * 60)

#     print("\nQuestion:")
#     print(question)

#     print("\nTop semantic matches:\n")

#     results = retrieve_top_k_chunks(question, top_k=3)

#     for idx, res in enumerate(results, start=1):
#         print(f"Result {idx}")
#         print("-" * 40)
#         print(f"Document   : {res['document']}")
#         print(f"Chunk ID   : {res['chunk_id']}")
#         print(f"Similarity : {res['similarity']:.4f}")
#         print("Chunk Text :")
#         print(res['text'])
#         print()

#     print("=" * 60)
#     print("If results are conceptually relevant but not keyword-matched,")
#     print("semantic search is WORKING correctly.")
#     print("=" * 60)

# if __name__ == "__main__":
#     main()

