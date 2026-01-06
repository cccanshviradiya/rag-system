import sqlite3

conn = sqlite3.connect("data/documents.db")
cursor = conn.cursor()

cursor.execute("SELECT document_name, chunk_id, chunk_text FROM documents")
rows = cursor.fetchall()

for row in rows:
    print("\n---")
    print("Document:", row[0])
    print("Chunk ID:", row[1])
    print("Text:", row[2][:150], "...")

conn.close()
