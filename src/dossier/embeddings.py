from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from .config import Settings, get_settings


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder(Embedder):
    """Deterministic hashed bag-of-words embedder.

    No model download, so tests and CI stay fast. Similarity is lexical only,
    which is enough to exercise retrieval, caching and access control logic.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        from .bm25_index import tokenize  # shares stopword/accent normalisation with BM25

        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class SentenceTransformerEmbedder(Embedder):
    """Multilingual dense embeddings. E5 models expect 'query:' / 'passage:' prefixes."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self._is_e5 = "e5" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_e5:
            text = f"query: {text}"
        return self.model.encode(text, normalize_embeddings=True)


class OllamaEmbedder(Embedder):
    """Embeddings served by Ollama (e.g. bge-m3, nomic-embed-text). No torch install needed."""

    def __init__(self, url: str, model: str, timeout: float = 120.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self._is_nomic = "nomic" in model.lower()
        self.dim = int(self._embed(["dimension probe"]).shape[1])

    def _embed(self, texts: list[str]) -> np.ndarray:
        from .http_retry import post_with_retry

        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            r = post_with_retry(
                f"{self.url}/api/embed", {"model": self.model, "input": texts[i : i + 32]}, self.timeout
            )
            out.extend(r.json()["embeddings"])
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._is_nomic:
            texts = [f"search_document: {t}" for t in texts]
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._is_nomic:
            text = f"search_query: {text}"
        return self._embed([text])[0]


def build_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    if settings.embedding_backend == "ollama":
        return OllamaEmbedder(settings.ollama_url, settings.ollama_embedding_model, settings.llm_timeout)
    return SentenceTransformerEmbedder(settings.embedding_model)
