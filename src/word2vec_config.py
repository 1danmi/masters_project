"""Configuration settings for the Word2Vec training pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import config


class Word2VecConfig(BaseSettings):
    """Pydantic settings controlling the Word2Vec training script."""

    model_config = SettingsConfigDict(env_prefix="WORD2VEC_")

    db_path: Path = Field(default_factory=lambda: Path(config().disambiguated_db_path))
    table: str = Field(default_factory=lambda: config().results_table)
    text_column: str = Field(default_factory=lambda: config().disambiguated_column)
    pk_column: str = Field(default_factory=lambda: config().index_columns)
    batch_size: int = 50_000
    workers: int = Field(default_factory=lambda: config().workers_count)

    vector_size: int = 100
    window: int = 5
    min_count: int = 5
    sample: float = 1e-3
    negative: int = 5
    epochs: int = 5
    sg: bool = False
    lowercase: bool = False
    strip: bool = True

    output_model: Path = Path("models/disambiguated.w2v")
    vectors_output: Path | None = None
    log_level: str = "INFO"
    compute_loss: bool = True


@lru_cache
def word2vec_config() -> Word2VecConfig:
    """Return a cached instance of :class:`Word2VecConfig`."""

    return Word2VecConfig()


__all__ = ["Word2VecConfig", "word2vec_config"]
