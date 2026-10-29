"""
src/config.py — All settings in one place.

Defaults live here. Any value can be overridden from .env.
Secrets (NVIDIA_API_KEY) must come from .env.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    DOCS_DIR: Path = Field(default=Path("Input_docs"))

    # Qdrant
    QDRANT_URL: str = Field(default="http://localhost:6333")
    COLLECTION_NAME: str = Field(default="medical_docs")

    # Parsing
    OCR_ENABLED: bool = Field(default=True)
    OCR_LANGUAGE: str = Field(default="eng")

    # Chunking (all sizes in characters, not tokens)
    MAX_CHUNK_CHARS: int = Field(default=1500)
    MIN_CHUNK_CHARS: int = Field(default=300)
    CHUNK_OVERLAP_CHARS: int = Field(default=200)

    # Embedding
    DENSE_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    DENSE_DIM: int = Field(default=384)
    SPARSE_MODEL: str = Field(default="Qdrant/bm25")

    # Retrieval
    SEARCH_MODE: str = Field(default="hybrid")   # hybrid | dense | sparse
    TOP_K: int = Field(default=5)
    CANDIDATES_PER_BRANCH: int = Field(default=20)
    FUSION: str = Field(default="rrf")

    # LLM
    LLM_MODEL: str = Field(default="nvidia/nemotron-ultra-253b-v1")
    LLM_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")
    TEMPERATURE: float = Field(default=0.2)
    MAX_TOKENS: int = Field(default=2048)

    # Secret — comes from .env only
    NVIDIA_API_KEY: str = Field(default="")


# Import this everywhere:  from src.config import settings
settings = Settings()
