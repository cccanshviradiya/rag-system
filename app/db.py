import sqlite3

DB_PATH = "data/documents.db"

def get_connection():
    return sqlite3.connect(DB_PATH)


## create table 
def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT,
            chunk_id INTEGER,
            chunk_text TEXT,
            embedding BLOB
                   
        )
    """)

    conn.commit()
    conn.close()
