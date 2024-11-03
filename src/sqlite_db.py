import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = HERE / "db.sqlite"


def create_connection():
    conn = sqlite3.connect(str(DB_PATH))
    return conn


def create_chunks_table(conn):
    cursor = conn.cursor()
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        language TEXT NOT NULL,
        pinecone_index_name TEXT NOT NULL,
        pinecone_namespace TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        source_title TEXT NOT NULL,
        source_url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )
    conn.commit()


def create_db():
    conn = create_connection()
    create_chunks_table(conn)
    conn.close()
    print(f"Database created successfully at {DB_PATH}")


if __name__ == "__main__":
    create_db()
