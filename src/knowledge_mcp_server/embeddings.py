import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class Embeddings:
    """Small wrapper around sentence-transformers for local embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Return a list of vectors for input texts."""
        return self.model.encode(texts, convert_to_numpy=True)
