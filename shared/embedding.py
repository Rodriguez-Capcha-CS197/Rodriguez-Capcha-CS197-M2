"""Shared embedding wrappers used by paper jobs."""

from __future__ import annotations

import numpy as np


class MiniLMEmbedder:
    """Thin SentenceTransformer wrapper with the embed/embed_single interface."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
