"""Sentence embeddings, behind a Protocol -- same pattern as `Transcriber`
and `ChatModel` in the main package, so metric functions and their tests
never need to load the real (large, slow-to-load) embedding model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return one embedding vector per input string, shape (n, dim)."""
        ...


class SentenceTransformerEmbedder:
    """Multilingual MiniLM via sentence-transformers.

    Loaded lazily and cached on the instance: constructing this class is
    free, loading the model (a few hundred MB) is not, and a benchmark run
    calls `embed` many times but only needs the model loaded once.
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return np.asarray(model.encode(texts, normalize_embeddings=True))
