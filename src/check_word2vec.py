"""Utilities for exploring Word2Vec analogies.

This module exposes helpers for loading a trained Word2Vec model and
experimenting with vector arithmetic such as the classic ``king - man +
woman`` analogy.  It can list the closest words to the resulting vector as
well as report how similar a specific word is to the analogy result and how
many words rank above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.word2vec_config import word2vec_config


def _token_sign(token: str) -> tuple[int, str]:
    """Return the sign (+1 or -1) and the associated word for a token.

    Tokens may optionally start with ``+`` or ``-`` (for example ``"+king"`` or
    ``"-man"``).  A token without a prefix is interpreted as positive.
    """

    if not token:
        raise ValueError("Empty token is not allowed")

    if token[0] == "+":
        return 1, token[1:]
    if token[0] == "-":
        return -1, token[1:]
    return 1, token


def expression_vector(kv: KeyedVectors, tokens: Sequence[str]) -> np.ndarray:
    """Compute the vector for a sequence of add/subtract tokens.

    Parameters
    ----------
    kv:
        The keyed vectors extracted from the Word2Vec model.
    tokens:
        A sequence such as ``["king", "-man", "+woman"]``.  Tokens may start
        with ``+`` or ``-`` to denote addition or subtraction.  The first token
        may omit the ``+`` sign.

    Returns
    -------
    numpy.ndarray
        The resulting vector representing the linear combination of the
        requested word vectors.
    """

    if not tokens:
        raise ValueError("At least one token is required to build an expression")

    result: np.ndarray | None = None
    for token in tokens:
        sign, word = _token_sign(token)
        if not word:
            raise ValueError(f"Token '{token}' does not contain a word")
        if word not in kv:
            raise KeyError(f"Word '{word}' is not in the model's vocabulary")

        vector = kv.get_vector(word)
        if result is None:
            result = sign * vector.astype(np.float64)
        else:
            result += sign * vector

    if result is None:
        raise RuntimeError("Failed to build the expression vector")

    return result


def _expression_words(tokens: Iterable[str]) -> set[str]:
    """Return the set of distinct words referenced by ``tokens``."""

    words: set[str] = set()
    for token in tokens:
        _, word = _token_sign(token)
        if word:
            words.add(word)
    return words


def closest_words(
    kv: KeyedVectors,
    vector: np.ndarray,
    topn: int,
    *,
    exclude: Iterable[str] | None = None,
) -> list[tuple[str, float, float]]:
    """Return the top-N closest words to ``vector``.

    The return value is a list of tuples ``(word, similarity, distance)`` where
    ``distance`` is defined as ``1 - similarity``.
    """

    if topn <= 0 or len(kv) == 0:
        return []

    excluded = {word for word in (exclude or ()) if word in kv}
    fetch = min(len(kv), topn + len(excluded) or 1)
    closest: list[tuple[str, float]] = []

    while True:
        candidates = kv.similar_by_vector(vector, topn=fetch)
        filtered = [(word, similarity) for word, similarity in candidates if word not in excluded]

        if len(filtered) >= topn or fetch == len(kv):
            closest = filtered[:topn]
            break

        # Increase the fetch size in an attempt to collect enough candidates.
        new_fetch = min(len(kv), max(fetch + len(excluded), fetch * 2))
        if new_fetch == fetch:
            closest = filtered
            break
        fetch = new_fetch

    return [(word, similarity, 1.0 - similarity) for word, similarity in closest]


def rank_word(kv: KeyedVectors, vector: np.ndarray, word: str) -> tuple[float, int, int]:
    """Return similarity metrics for ``word`` relative to ``vector``.

    Returns a tuple ``(similarity, rank, more_similar_count)``.  ``rank`` is
    1-based, therefore a rank of 1 means the target word is the closest match to
    ``vector`` across the whole vocabulary.
    """

    if word not in kv:
        raise KeyError(f"Word '{word}' is not in the model's vocabulary")

    similarities = kv.cosine_similarities(vector, kv.vectors)
    index = kv.key_to_index[word]
    similarity = float(similarities[index])
    more_similar = int(np.sum(similarities > similarity))
    rank = more_similar + 1
    return similarity, rank, more_similar


def load_model(path: Path) -> KeyedVectors:
    """Load the Word2Vec model from ``path`` and return its keyed vectors."""

    model = Word2Vec.load(str(path))
    return model.wv


@dataclass(slots=True)
class ClosestWord:
    """Representation of a word closest to an analogy expression."""

    word: str
    similarity: float
    distance: float


@dataclass(slots=True)
class RankedWord:
    """Representation of a ranked comparison word."""

    word: str
    similarity: float
    rank: int
    more_similar_count: int


@dataclass(slots=True)
class AnalogyEvaluation:
    """Container for the results of evaluating an analogy expression."""

    expression: tuple[str, ...]
    vector: np.ndarray
    closest: list[ClosestWord]
    ranked_words: list[RankedWord]


class CheckWord2VecConfig(BaseSettings):
    """Settings controlling the Word2Vec analogy exploration helpers."""

    model_config = SettingsConfigDict(env_prefix="CHECK_WORD2VEC_")

    model_path: Path = Field(default_factory=lambda: word2vec_config().output_model)
    tokens: tuple[str, ...] = Field(default_factory=tuple)
    topn: int = 10
    rank_words: tuple[str, ...] = Field(default_factory=tuple)
    exclude_expression_words: bool = True


@lru_cache
def check_word2vec_config() -> "CheckWord2VecConfig":
    """Return a cached instance of :class:`CheckWord2VecConfig`."""

    return CheckWord2VecConfig()


def evaluate_expression(
    kv: KeyedVectors,
    tokens: Sequence[str],
    *,
    topn: int = 10,
    rank_words: Sequence[str] = (),
    exclude_expression_words: bool = True,
) -> AnalogyEvaluation:
    """Evaluate an analogy expression using the provided keyed vectors."""

    vector = expression_vector(kv, tokens)
    excluded_words = _expression_words(tokens) if exclude_expression_words else set()

    closest = [
        ClosestWord(word=word, similarity=similarity, distance=distance)
        for word, similarity, distance in closest_words(
            kv,
            vector,
            topn,
            exclude=excluded_words,
        )
    ]

    ranked = [
        RankedWord(word=word, similarity=similarity, rank=rank, more_similar_count=more_similar)
        for word in rank_words
        for similarity, rank, more_similar in (rank_word(kv, vector, word),)
    ]

    return AnalogyEvaluation(
        expression=tuple(tokens),
        vector=vector,
        closest=closest,
        ranked_words=ranked,
    )


def evaluate_from_settings(settings: CheckWord2VecConfig | None = None) -> AnalogyEvaluation:
    """Evaluate an analogy expression using the provided settings object."""

    settings = settings or check_word2vec_config()

    if not settings.tokens:
        raise ValueError("At least one token must be supplied in the settings")

    kv = load_model(settings.model_path)
    return evaluate_expression(
        kv,
        settings.tokens,
        topn=settings.topn,
        rank_words=settings.rank_words,
        exclude_expression_words=settings.exclude_expression_words,
    )

