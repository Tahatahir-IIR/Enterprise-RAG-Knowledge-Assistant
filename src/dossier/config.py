from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_backend: Literal["ollama", "openai", "stub"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_timeout: float = 120.0

    # Embeddings
    embedding_backend: Literal["sentence-transformers", "ollama", "hash"] = "sentence-transformers"
    embedding_model: str = "intfloat/multilingual-e5-small"
    ollama_embedding_model: str = "bge-m3"

    # Vector store
    qdrant_url: str = ""
    qdrant_path: str = "data/index/qdrant"
    chunks_collection: str = "chunks"
    cache_collection: str = "semantic_cache"

    # Retrieval
    chunk_strategy: Literal["fixed", "recursive", "semantic"] = "recursive"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    candidates: int = 20
    # Tune per embedding model with scripts/evaluate.py (refusal vs. answered metrics).
    # bge-m3: relevant ~0.6+, off-topic ~0.4. e5: everything scores high, rely on NOT_FOUND.
    min_dense_score: float = 0.50
    rrf_k: int = 60

    # Cache
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95

    # Access control / storage
    users_file: str = "config/users.yaml"
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def index_path(self) -> Path:
        return self.data_path / "index"


@lru_cache
def get_settings() -> Settings:
    return Settings()
