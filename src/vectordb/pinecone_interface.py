from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
import yaml

from vectordb.chunk_utils import Chunk

load_dotenv()

HERE = Path(__file__).parent
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)


def create_indices():
    # Load settings from YAML file
    settings_path = HERE / ".." / "settings.yaml"
    with open(settings_path, "r") as file:
        settings = yaml.safe_load(file)

    pinecone_settings = settings["pinecone"]
    # Create indices for each language
    for lang, config in pinecone_settings["indices"].items():
        index_name = config["index_name"]
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=config["dimension"],
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=pinecone_settings["cloud"], region=pinecone_settings["region"]
                ),
            )
            print(f"Created index: {index_name}")
        else:
            print(f"Index {index_name} already exists")


def insert_vectors(chunks: list[Chunk], index_name: str, namespace: str):
    """
    chunks is a list of dicts with the following structure:
    {
        "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "metadata": {
            "source_title": "Title",
            "source_url": "URL",
            "chunk_index": 0,
            "chunk_text": "Text",
            "language": "en"
        }
    }
    """
    data = []
    for chunk in chunks:
        data.append(
            {
                "id": chunk.id,
                "values": chunk.vector,
                "metadata": {
                    "source_title": chunk.source_title,
                    "chunk_index": chunk.chunk_index,
                },
            }
        )

    index = pc.Index(index_name)
    index.upsert(
        vectors=data,
        namespace=namespace,
    )
