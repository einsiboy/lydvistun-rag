from pathlib import Path
import sqlite3

from vectordb.chunk_utils import Chunk

HERE = Path(__file__).parent
DB_PATH = HERE / "db.sqlite"


def delete_chunks(ids: list[str]):
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?" for _ in ids])
        cursor.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
        conn.commit()


def insert_chunks(data: list[Chunk]):
    bindvars = []
    for chunk in data:
        bindvars.append(
            (
                chunk.id,
                chunk.language,
                chunk.pinecone_index_name,
                chunk.pinecone_namespace,
                chunk.chunk_text,
                chunk.chunk_index,
                chunk.source_title,
                chunk.source_url,
            )
        )

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            """
        INSERT INTO chunks (
            id,
            language,
            pinecone_index_name,
            pinecone_namespace,
            chunk_text,
            chunk_index,
            source_title,
            source_url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            bindvars,
        )
        conn.commit()


def get_chunks(ids: list[str]) -> list[Chunk]:
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?" for _ in ids])
        cursor.execute(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)
        rows = cursor.fetchall()

        chunks = []
        for row in rows:
            chunks.append(
                Chunk(
                    id=row[0],
                    language=row[1],
                    pinecone_index_name=row[2],
                    pinecone_namespace=row[3],
                    chunk_text=row[4],
                    chunk_index=row[5],
                    source_title=row[6],
                    source_url=row[7],
                )
            )
        return chunks
