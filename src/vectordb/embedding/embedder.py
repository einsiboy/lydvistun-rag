from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

HERE = Path(__file__).parent


class Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct"):
        # local_model_path = f"./models/{model_name.replace('/', '_')}"
        local_model_path = HERE / "models" / f"{model_name.replace('/', '_')}"
        local_model_path = str(local_model_path)

        if os.path.exists(local_model_path):
            self.model = SentenceTransformer(local_model_path)
        else:
            self.model = SentenceTransformer(model_name)
            os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
            self.model.save(local_model_path)

    def embed(self, texts):
        return self.model.encode(
            texts,
            batch_size=64,
            precision="float32",
            normalize_embeddings=True,
            convert_to_numpy=True,
            # convert_to_tensor=True
        )
