from collections import Counter
from typing import Any

import numpy as np
from pydantic import BaseModel


class TokenEntry(BaseModel):
    bow: dict[str, int] = {}  # Environment of the token, each entry is a {word: count}
    bow_b2v: dict[str, int] = {}
    token: str  # The string representation of the token
    vec: np.ndarray  # The vector representation of the token
    count: int = 1  # Counter for the token
    token_id: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @staticmethod
    def weighted_vector_sum(vectors_count: list[tuple[np.ndarray, int]]) -> np.ndarray:
        total_count = sum([t[1] for t in vectors_count])  # Total count of all vectors
        # First, multiply each vector by its relative count (t[1]/total_count).
        arr = np.array(
            [np.multiply(t[0], t[1] / total_count) for t in vectors_count]  # t[0] is the vector, t[1] is the count
        )  # Shape (len(vectors_count), vector_dimension)

        # Add all vectors across rows, i.e., sum each column.
        return arr.sum(axis=0)

    def update_vec_count(self, new_vec: np.ndarray, count: int = 1):
        # Combine the current vector with a new vector, weighted by how many times each was seen.
        self.vec = self.weighted_vector_sum([(self.vec, self.count), (new_vec, count)])
        self.count = self.count + count

    def update_bow(self, bow: list[str], bow_b2v: list[str]):
        for token in bow:
            self.bow[token] = self.bow.get(token, 0) + 1
        for token in bow_b2v:
            self.bow_b2v[token] = self.bow_b2v.get(token, 0) + 1

    def unite_entries(self, entry: "TokenEntry"):
        self.update_vec_count(entry.vec, entry.count)
        self.bow = dict(Counter(self.bow) + Counter(entry.bow))
        self.bow_b2v = dict(Counter(self.bow_b2v) + Counter(entry.bow_b2v))

    def __repr__(self):
        return f"{self.token!r},{self.vec!r},{self.bow!r},{self.count!r}"


type TokenEntries = list[TokenEntry]

type Embeddings = dict[str, TokenEntries]
