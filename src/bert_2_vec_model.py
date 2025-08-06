import os
import logging
import shelve
from pathlib import Path
from typing import Self, Final

from src.utils.write_behind_shelve import WriteBehindShelf

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from src.data_models import Embeddings, TokenEntry, TokenEntries
from src.utils.tf_utils import get_bow_idf_dict
from src.utils.parsing_utils import tokenize_sentence, get_bow
from src.utils.models_utils import BERT_VECTOR_SIZE, get_bert_vec

ACCEPT_THRESHOLD: Final[float] = config().accept_threshold
RADIUS: Final[float] = config().radius

logger = logging.getLogger(__name__)


class Bert2VecModel:

    def __init__(
        self,
        source_path: Path | str,
        dest_path: Path | str | None = None,
        in_mem: bool = False,
        new_model: bool = False,
        cache_size: int | None = None,
        flush_interval_seconds: float | None = None,
    ):
        self._source_path = Path(source_path)
        self._dest_path = Path(dest_path or source_path)
        self._embeddings: Embeddings | shelve.Shelf[TokenEntries] | WriteBehindShelf | None = None
        self._cache_size = cache_size or config().write_cache_max_size
        self._flush_interval_seconds = flush_interval_seconds or config().write_cache_flush_seconds
        if new_model:
            for filename in os.listdir(self._dest_path.parent):
                file_path = os.path.join(self._dest_path.parent, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        self.load_data(in_mem=in_mem)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def get_shelve_path(path: Path | str) -> Path:
        path = Path(path)
        if path.suffix != ".dat":
            path = path.with_name(path.name + ".dat")
        return path

    def _verify_source_table_exist(self) -> str:
        path = self.get_shelve_path(self._source_path)
        # if not path.is_file():
        #     raise FileNotFoundError(f"File '{self._source_path}' doesn't exist")
        return str(self._source_path)

    def load_data(self, in_mem=False) -> None:
        print("Loading model...")
        source_file_path = self._verify_source_table_exist()
        if not in_mem:
            self._embeddings = WriteBehindShelf(
                source_file_path,
                max_cache_size=self._cache_size,
                flush_interval=self._flush_interval_seconds,
            )
            print("Finished loading model...")
        else:
            with shelve.open(source_file_path) as s:
                self._embeddings = dict(s)  # read whole
            print(f"Finished loading model with {len(self._embeddings)}...")

    def save_data(self) -> None:
        if isinstance(self._embeddings, dict):
            if not os.path.exists(self._dest_path.parent):
                os.makedirs(self._dest_path.parent)
            with shelve.open(self._source_path) as s:
                s.update(self._embeddings)
                s.sync()
        else:
            self._embeddings.sync()

    def close(self):
        self.save_data()
        if hasattr(self._embeddings, "close"):
            self._embeddings.close()

    def __getitem__(self, token: str):
        if isinstance(token, str):
            if self._embeddings is None:
                raise RuntimeError("Embeddings are not loaded. Please load the model first.")
            return self._embeddings.get(token)
        else:
            raise TypeError(f"Unsupported type for Bert2VecModel: {type(token)}")

    def get_most_frequent_vec(self, token: str):
        entries: TokenEntries = self._embeddings.get(token)
        match len(entries):
            case 0:
                return np.zeros(BERT_VECTOR_SIZE)
            case 1:
                return entries[0].vec
            case _:
                return max(entries, key=lambda entry: entry.count).vec

    def get_entries_by_bert(
        self, token: str, sentence: str, max_results: int | None = None
    ) -> list[tuple[TokenEntry, float]]:
        bert_vector = get_bert_vec(token=token, sentence=sentence)
        results = []
        # We want to find calculate how similar are our embeddings to the BERT embedding we got.
        if token in self._embeddings:
            for row in self._embeddings[token]:
                similarity = cosine_similarity([bert_vector], [row.vec])[0][0]
                results.append((row, similarity))

        # Sort the results by their similarity (the most similar first).
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        return sorted_results[:max_results] if max_results else sorted_results

    def get_entries_by_bow_bm25(
        self, token: str, bow: list[str], k1: float = 1.5, max_results: int | None = None
    ) -> list[tuple[TokenEntry, float]]:
        # if token not in bow:
        #     raise ValueError(f"Token '{token}' must be in the bag of words (BOW).")

        entries = self._embeddings.get(token)
        if not entries:
            return []

        bow_idf_dict = get_bow_idf_dict(bow, entries)

        bm25 = []
        for entry in entries:
            bm25 += [
                (
                    entry,
                    sum(
                        bow_idf_dict.get(w)  # IDF
                        * (
                            (entry.bow.get(w, 0) * (k1 + 1)) / (entry.bow.get(w, 0) + k1)
                        )  # * (1 - b + b * (sum(r.BOW.values()) / avgdl))))
                        for w in bow
                    ),
                )
            ]

        results = sorted(bm25, key=lambda t: t[1], reverse=True)
        return results[:max_results] if max_results else results

    def get_entry_by_sentence(self, token: str, sentence: str):
        tokens = tokenize_sentence(sentence)
        token_idx = tokens.index(token)
        bow = get_bow(tokens=tokens, idx=token_idx)
        return self.get_entry_by_bow(token=token, bow=bow)

    def get_entry_by_bow(self, token: str, bow: list[str]) -> TokenEntry:
        result = self.get_entries_by_bow_bm25(token=token, bow=bow, max_results=1)
        return result[0][0] if result else None

    def get_entry_by_vec(self, token: str, vec: np.ndarray) -> tuple[TokenEntry | None, float]:
        entries = self._embeddings.get(token, [])
        closest_entry = None
        max_cos = -1
        for entry in entries:
            cos: float = cosine_similarity([entry.vec], [vec])[0][0]
            if cos > max_cos:
                max_cos = cos
                closest_entry = entry
        return closest_entry, max_cos

    def add_entry(self, entry: TokenEntry) -> None:
        closest_entry, max_similarity = self.get_entry_by_vec(token=entry.token, vec=entry.vec)

        entries = self._embeddings.get(entry.token, [])
        if closest_entry is not None and max_similarity > ACCEPT_THRESHOLD:
            closest_entry.update_vec_count(new_vec=entry.vec)
            closest_entry.update_bow(bow=entry.bow, bow_b2v=entry.bow_b2v)
        else:
            entries.append(entry)
        self._embeddings[entry.token] = entries  # ensure persistence
