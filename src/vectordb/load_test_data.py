import json
from .embedding.embedder import Embedder
from .pinecone_interface import insert_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_text(text)


def load_and_embed_test_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    """Should be of the form:
      vectors=[
    {
      "id": "A", 
      "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
      "metadata": {"genre": "comedy", "year": 2020}
    },
    """
    chunks = []

    for i, item in enumerate(data):
        split_texts = split_text(item["main_text"])

        for j, chunk in enumerate(split_texts):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "title": item["name"],
                        "url": item["url"],
                        "chunk_index": j,
                    },
                }
            )

    embedder = Embedder()
    vectors = embedder.embed(chunks)

    for chunk, vector in zip(chunks, vectors):
        chunk["values"] = vector

    return chunks


def insert_test_data(index_name, vectors, metadata):
    insert_vectors(index_name, vectors, metadata)


def main():
    file_path = "data/visindavefur_articles.json"
    vectors = load_and_embed_test_data(file_path)

    namespace = "visindavefur"
    insert_test_data("is-index", vectors, namespace)


if __name__ == "__main__":
    main()
