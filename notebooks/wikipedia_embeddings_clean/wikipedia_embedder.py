import logging
import os
from typing import Dict, List

import pandas as pd
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WikiEmbedder:
    def __init__(
        self,
        input_parquet_path: str,
        checkpoint_path: str,
        pinecone_index_name: str,
        batch_size: int = 50,
        embedding_model: str = "multilingual-e5-large",
    ):
        self.input_parquet_path = input_parquet_path
        self.checkpoint_path = checkpoint_path
        self.index_name = pinecone_index_name
        self.batch_size = batch_size
        self.embedding_model = embedding_model

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(self.index_name)

        # Load or create checkpoint
        self.processed_chunks = self._load_checkpoint()

        self.df = pd.read_parquet(self.input_parquet_path)

        # Normalize titles, needed for vector id and pinecone needs ascii
        self.df["title_normalized"] = (
            self.df.title.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
            .str.replace(r"[^a-z0-9_-]", "", regex=True)
        )

        self.df_to_process = self.get_unprocessed_articles()

    def get_unprocessed_articles(self) -> pd.DataFrame:
        """Get articles that have at least one unprocessed chunk"""

        # Create a set of (title_match, chunk_idx) tuples for faster lookup
        processed_chunks = set(
            zip(
                self.processed_chunks["title_match"], self.processed_chunks["chunk_idx"]
            )
        )

        # Create a mask for unprocessed chunks
        def get_unprocessed_mask(row):
            return any(
                (row["title_match"], chunk_idx) not in processed_chunks
                for chunk_idx in range(row["n_chunks"])
            )

        # Filter to articles that have at least one unprocessed chunk
        df_to_process = self.df[self.df.apply(get_unprocessed_mask, axis=1)].copy()

        logging.info(
            f"Found {len(self.df) - len(df_to_process)} fully processed articles"
        )
        logging.info(
            f"Processing remaining {len(df_to_process)} articles (including partially processed)"
        )

        return df_to_process

    def _load_checkpoint(self) -> pd.DataFrame:
        """Load existing checkpoint or create empty DataFrame"""
        if os.path.exists(self.checkpoint_path):
            return pd.read_parquet(self.checkpoint_path)
        return pd.DataFrame(columns=["title_match", "chunk_idx"])

    def _save_checkpoint(self, new_chunks: pd.DataFrame):
        """Append new chunks to checkpoint and save"""
        self.processed_chunks = pd.concat([self.processed_chunks, new_chunks])
        self.processed_chunks.to_parquet(self.checkpoint_path)

    def prepare_chunks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare chunks for embedding with proper formatting"""
        prefix_template = (
            "This is text from the wikipedia article with title: '{title}'"
        )
        data = []

        for idx, row in df.iterrows():
            for i in range(row["n_chunks"]):
                # Skip if chunk already processed
                if self._is_processed(row["title_match"], i):
                    continue

                chunk_text = row["split_text"][i]
                prefix = prefix_template.format(title=row["title"].strip())
                chunk_text = f"{prefix}\n{chunk_text}"

                data.append(
                    {
                        "chunk_idx": i,
                        "chunk_text": chunk_text,
                        "title_match": row["title_match"],
                        "title": row["title"].strip(),
                        "url": row["url"],
                        "title_normalized": row["title_normalized"],
                        "word_count_total": row["word_count"],
                        "word_count_chunk": row["n_words_per_chunk"][i],
                        "chunk_size": row["chunk_size"],
                    }
                )

        return pd.DataFrame(data)

    def _is_processed(self, title_match: str, chunk_idx: int) -> bool:
        """Check if chunk has already been processed"""
        return (
            len(
                self.processed_chunks[
                    (self.processed_chunks.title_match == title_match)
                    & (self.processed_chunks.chunk_idx == chunk_idx)
                ]
            )
            > 0
        )

    def embed_chunks(self, df_to_embed: pd.DataFrame):
        """Generate embeddings in batches"""
        all_embeddings = []

        for start_idx in range(0, len(df_to_embed), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_texts = df_to_embed["chunk_text"].iloc[start_idx:end_idx].tolist()

            batch_embeddings = self.pc.inference.embed(
                model=self.embedding_model,
                inputs=batch_texts,
                parameters={"input_type": "passage", "truncate": "END"},
            )

            all_embeddings.extend(batch_embeddings)
            logging.info(
                f"Processed batch {start_idx // self.batch_size + 1}: Embeddings {start_idx} to {end_idx - 1}"
            )

        return [e["values"] for e in all_embeddings]

    def prepare_vector_data(self, df_to_embed: pd.DataFrame) -> List[Dict]:
        """Prepare data for Pinecone insertion"""
        df_to_embed["vector_id"] = (
            df_to_embed["title_normalized"] + "_" + df_to_embed["chunk_idx"].astype(str)
        )

        vector_data = []
        for idx, row in df_to_embed.iterrows():
            vector_data.append(
                {
                    "id": row["vector_id"],
                    "values": row["embedding_vector"],
                    "metadata": {
                        "title": row["title"],
                        "url": row["url"],
                        "chunk_idx": row["chunk_idx"],
                        "chunk_text": row["chunk_text"],
                    },
                }
            )
        return vector_data

    def process_articles(self, articles_per_batch: int = 1000):
        """
        Main processing function
        """
        # Process articles in batches
        for start_idx in range(0, len(self.df_to_process), articles_per_batch):
            df_batch = self.df_to_process.iloc[
                start_idx : start_idx + articles_per_batch
            ]
            logging.info(
                f"Processing articles {start_idx} to {start_idx + len(df_batch) - 1}"
            )
            logging.info(f"N chunks in batch: {df_batch['n_chunks'].sum()}")

            # Prepare chunks for embedding
            df_to_embed = self.prepare_chunks(df_batch)

            if len(df_to_embed) == 0:
                logging.info("No new chunks to process in this batch")
                continue

            # Generate embeddings
            embeddings = self.embed_chunks(df_to_embed)
            df_to_embed["embedding_vector"] = embeddings

            # Prepare and upsert to Pinecone
            vector_data = self.prepare_vector_data(df_to_embed)
            for i in range(0, len(vector_data), self.batch_size):
                batch = vector_data[i : i + self.batch_size]
                self.index.upsert(vectors=batch)

            # Update checkpointdd
            checkpoint_data = df_to_embed[["title_match", "chunk_idx"]]
            self._save_checkpoint(checkpoint_data)

            logging.info(
                f"Completed processing batch. Total processed chunks: {len(self.processed_chunks)}"
            )

    def print_init_info(self):
        print()
        print("=" * 100)
        print(f"Original number of articles: {len(self.df)}")
        print(f"Total number of chunks: {self.df['n_chunks'].sum()}")
        print(f"Embedding model: {self.embedding_model}")
        print(f"Batch size: {self.batch_size}")
        print(f"Checkpoint path: {self.checkpoint_path}")
        print(f"N rows in checkpoint: {len(self.processed_chunks)}")
        print(
            f"N unique articles in checkpoint: {len(self.processed_chunks['title_match'].unique())}"
        )
        print(f"N articles to process: {len(self.df_to_process)}")
        print(
            "n checkpoint + n top process (sanity check):",
            len(self.processed_chunks["title_match"].unique())
            + len(self.df_to_process),
        )
