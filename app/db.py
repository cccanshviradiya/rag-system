import sqlite3

DB_PATH = "data/documents.db"


def get_connection():
    """
    Create and return a SQLite database connection.
    """
    conn = sqlite3.connect(DB_PATH)
    return conn


def create_table():
    """
    Create documents table if it does not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT,
            chunk_id INTEGER,
            chunk_text TEXT,
            embedding BLOB
        )
        """
    )

    conn.commit()
    conn.close()
