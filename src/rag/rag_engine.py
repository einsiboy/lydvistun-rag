from vectordb.embedding.embedder import Embedder
from vectordb.pinecone_interface import pc
import datastore
import together
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

load_dotenv()


class ResponseWithCitation(BaseModel):
    response: str = Field(description="The full response from the LLM")
    source_ids: list[str] = Field(
        description="A list of the source_id ids used by the LLM in the response, not the text itself"
    )
    highlighted_response: list[dict[str, str]] = Field(
        description="""A list of dictionaries. Each dictionary has a key and a value: 
        key: source_id, which is the id of the source that the text is from,
        value: a substring of the response that is highlighted to indicate where the citation is used
        """,
        example=[
            {"source_id_1: Prince Charles was often seen hunting fox"},
            {
                "source_id_2": "When the Queen visited the USA, she was given a pair of diamond earrings",
            },
        ],
    )


class RAGEngine:

    system_message_template = """
You are a helpful assistant.

You are given a message from a user and a list of citations.
If the citations are relevant to the message, you should use them to answer the question.

ONLY RESPOND IN JSON.

Please respond in the same language as the message from the user.
"""

    prompt_template = """
You will be given a message from a user and a list of sources, 
that may be relevant to answering the question.
Your job is to answer the question.

Here is the question:
{question}

Sources:
{sources}
"""

    def __init__(self):
        self.settings = self._load_settings()
        self.embedders = self._initialize_embedders()
        self.indices = self._initialize_indices()
        self.together_client = together.Together(
            api_key=os.environ.get("TOGETHER_API_KEY")
        )

    def _load_settings(self):
        settings_path = Path(__file__).parent.parent / "settings.yaml"
        with open(settings_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_embedders(self):
        embedders = {}
        for lang, config in self.settings["pinecone"]["indices"].items():
            # TODO fix yaml section for english
            if lang != "icelandic":
                continue
            embedders[lang] = Embedder(model_name=config["embedding_model"])
        return embedders

    def _initialize_indices(self):
        indices = {}
        for lang, config in self.settings["pinecone"]["indices"].items():
            # TODO fix yaml section for english
            if lang != "icelandic":
                continue
            indices[lang] = pc.Index(config["index_name"])
        return indices

    def query(self, question: str, language: str, namespace: str, top_k: int = 5):
        if language not in self.embedders or language not in self.indices:
            raise ValueError(f"Unsupported language: {language}")

        # 1. Embed the query
        query_embedding = self.embedders[language].embed([question])[0]

        # 2. Query Pinecone to get the closest matches
        results = self.indices[language].query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            namespace=namespace,
        )

        # 3. Get the chunk text from SQLite database
        chunk_ids = [match["id"] for match in results["matches"]]
        chunks = datastore.get_chunks(chunk_ids)

        # 4. Use Together AI for inference with the context
        single_source_template = """
        ```
        source_id: {source_id}
        source_text: {source_text}
        ```
        """

        formatted_sources = "\n\n".join(
            [
                single_source_template.format(
                    source_id=chunk.id, source_text=chunk.chunk_text
                )
                for chunk in chunks
            ]
        )
        user_message = self.prompt_template.format(
            question=question, sources=formatted_sources
        )

        messages = [
            {"role": "system", "content": self.system_message_template},
            {"role": "user", "content": user_message},
        ]

        response = self.together_client.chat.completions.create(
            model=self.settings["together_ai"]["model"],
            messages=messages,
            response_format={
                "type": "json_object",
                "schema": ResponseWithCitation.model_json_schema(),
            },
        )

        # response_object = ResponseWithCitation(**response.choices[0].message.content)
        response_object = ResponseWithCitation(
            **json.loads(response.choices[0].message.content)
        )

        return {
            "question": question,
            # "answer": response.choices[0].message.content,
            "answer": response_object,
            "chunks": chunks,
        }
