"""Configuration helpers for the Word2Vec analogy utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.word2vec_config import word2vec_config


class CheckWord2VecConfig(BaseSettings):
    """Settings controlling the Word2Vec analogy exploration helpers."""

    model_config = SettingsConfigDict(env_prefix="CHECK_WORD2VEC_")

    model_path: Path = Field(default_factory=lambda: word2vec_config().output_model)
    pretrained_name: str | None = None
    tokens: tuple[str, ...] = Field(default_factory=tuple)
    topn: int = 10
    rank_words: tuple[str, ...] = Field(default_factory=tuple)
    exclude_expression_words: bool = True


__all__ = ["CheckWord2VecConfig"]
