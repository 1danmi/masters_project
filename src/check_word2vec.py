"""Utilities for exploring Word2Vec analogies.

This module provides a small command line interface for loading a trained
Word2Vec model and experimenting with vector arithmetic such as the classic
``king - man + woman`` analogy.  It can list the closest words to the
resulting vector as well as report how similar a specific word is to the
analogy result and how many words rank above it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

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


def closest_words(kv: KeyedVectors, vector: np.ndarray, topn: int) -> list[tuple[str, float, float]]:
    """Return the top-N closest words to ``vector``.

    The return value is a list of tuples ``(word, similarity, distance)`` where
    ``distance`` is defined as ``1 - similarity``.
    """

    closest = kv.similar_by_vector(vector, topn=topn)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explore Word2Vec analogies")
    parser.add_argument(
        "tokens",
        nargs="+",
        help="Expression tokens such as 'king', '-man', '+woman'. The first token may omit the '+' sign.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=word2vec_config().output_model,
        help="Path to the trained Word2Vec model (default: %(default)s)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="Number of closest words to display (default: %(default)s)",
    )
    parser.add_argument(
        "--rank",
        dest="rank_words",
        nargs="*",
        default=(),
        help="Optional list of words to evaluate against the analogy result",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    kv = load_model(args.model)
    vector = expression_vector(kv, args.tokens)

    print(f"Loaded model with {len(kv):,} words from {args.model}")
    print("Expression:", " ".join(args.tokens))

    print(f"\nTop {args.topn} closest words:")
    for rank, (word, similarity, distance) in enumerate(closest_words(kv, vector, args.topn), start=1):
        print(f"{rank:>2}. {word:<20} similarity={similarity:.4f} distance={distance:.4f}")

    for word in args.rank_words:
        similarity, rank, more_similar = rank_word(kv, vector, word)
        print(
            f"\nWord '{word}' similarity: {similarity:.4f}\n"
            f"Rank: {rank} (there are {more_similar} words with a higher similarity)"
        )


if __name__ == "__main__":
    main()

