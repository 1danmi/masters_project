from pathlib import Path
import pickle
import os
from tqdm import tqdm
import gzip
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import logging

from sklearn.metrics.pairwise import cosine_similarity

from .bert_2_vec_model import Bert2VecModel
from .utils.parsing_utils import tokenize_sentence, get_bow
from .utils.models_utils import get_bert_vec


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompactTokenEntry:
    """Memory efficient representation of :class:`TokenEntry`.

    ``bow`` and ``bow_b2v`` store arrays of ``[token_id, count]`` pairs
    rather than Python dictionaries in order to significantly reduce
    per-entry overhead.
    """

    token_id: int
    vec: np.ndarray
    count: int
    bow: np.ndarray  # shape (n, 2) -> [token_id, count]
    bow_b2v: np.ndarray  # shape (n, 2) -> [token_id, count]


class CompactBert2VecModel:
    """In-memory compact representation of a :class:`Bert2VecModel`.

    The model only supports read-only operations for querying the
    existing entries by BOW, sentence or BERT embedding.  It is designed
    to hold all data in memory and avoid the large overhead of Python
    dictionaries present in :class:`TokenEntry`.
    """

    def __init__(
        self,
        embeddings: Dict[int, List[CompactTokenEntry]],
        id_to_token: List[str],
        token_to_id: Dict[str, int],
    ) -> None:
        self._embeddings = embeddings
        self._id_to_token = id_to_token
        self._token_to_id = token_to_id

    # ------------------------------------------------------------------
    # Construction utilities
    # ------------------------------------------------------------------
    @classmethod
    def from_bert2vec(cls, model: Bert2VecModel) -> "CompactBert2VecModel":
        """Convert a :class:`Bert2VecModel` into a compact in-memory model."""

        token_to_id: Dict[str, int] = {}
        id_to_token: List[str] = []

        def get_id(token: str) -> int:
            idx = token_to_id.get(token)
            if idx is None:
                idx = len(id_to_token)
                token_to_id[token] = idx
                id_to_token.append(token)
            return idx

        embeddings: Dict[int, List[CompactTokenEntry]] = {}
        total_tokens = len(model._embeddings)
        logger.info("Starting conversion of %d tokens", total_tokens)
        for idx, (token, entries) in enumerate(model._embeddings.items(), start=1):  # type: ignore[attr-defined]
            token_id = get_id(token)
            compact_entries: List[CompactTokenEntry] = []
            for entry in entries:
                bow_arr = np.array([[get_id(w), c] for w, c in entry.bow.items()], dtype=np.int32)
                bow_b2v_arr = np.array([[get_id(w), c] for w, c in entry.bow_b2v.items()], dtype=np.int32)
                compact_entries.append(
                    CompactTokenEntry(
                        token_id=token_id,
                        vec=entry.vec,
                        count=entry.count,
                        bow=bow_arr,
                        bow_b2v=bow_b2v_arr,
                    )
                )
            embeddings[token_id] = compact_entries
            if idx % max(1, total_tokens // 100) == 0:
                logger.info(
                    "Converted %d/%d tokens (%.1f%%)",
                    idx,
                    total_tokens,
                    (idx / total_tokens) * 100,
                )

        logger.info("Finished converting tokens")
        return cls(embeddings=embeddings, id_to_token=id_to_token, token_to_id=token_to_id)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def _token_ids(self, words: Iterable[str]) -> List[int]:
        return [self._token_to_id[w] for w in words if w in self._token_to_id]

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def get_entries_by_bert(
        self, token: str, sentence: str, max_results: int | None = None
    ) -> List[Tuple[CompactTokenEntry, float]]:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            return []
        bert_vector = get_bert_vec(token=token, sentence=sentence)
        results: List[Tuple[CompactTokenEntry, float]] = []
        for entry in self._embeddings.get(token_id, []):
            similarity = cosine_similarity([bert_vector], [entry.vec])[0][0]
            results.append((entry, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results] if max_results else results

    def get_entries_by_bow_bm25(
        self, token: str, bow: List[str], k1: float = 1.5, max_results: int | None = None
    ) -> List[Tuple[CompactTokenEntry, float]]:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            return []
        entries = self._embeddings.get(token_id, [])
        if not entries:
            return []

        bow_ids = self._token_ids(bow)
        N = len(entries)
        idf: Dict[int, float] = {}
        for wid in bow_ids:
            df = sum(1 for e in entries if wid in e.bow[:, 0])
            idf[wid] = 0.0 if df == 0 else float(np.log2(N / df))

        results: List[Tuple[CompactTokenEntry, float]] = []
        for entry in entries:
            bow_dict = {wid: cnt for wid, cnt in entry.bow}
            score = sum(
                idf.get(wid, 0.0) * ((bow_dict.get(wid, 0) * (k1 + 1)) / (bow_dict.get(wid, 0) + k1)) for wid in bow_ids
            )
            results.append((entry, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results] if max_results else results

    def get_entry_by_bow(self, token: str, bow: List[str]) -> CompactTokenEntry | None:
        results = self.get_entries_by_bow_bm25(token=token, bow=bow, max_results=1)
        return results[0][0] if results else None

    def get_entry_by_sentence(self, token: str, sentence: str) -> CompactTokenEntry | None:
        tokens = tokenize_sentence(sentence)
        token_idx = tokens.index(token)
        bow = get_bow(tokens=tokens, idx=token_idx)
        return self.get_entry_by_bow(token=token, bow=bow)

    def get_entry_by_vec(self, token: str, vec: np.ndarray) -> Tuple[CompactTokenEntry | None, float]:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            return None, -1.0
        entries = self._embeddings.get(token_id, [])
        best_entry: CompactTokenEntry | None = None
        max_cos = -1.0
        for entry in entries:
            cos = cosine_similarity([entry.vec], [vec])[0][0]
            if cos > max_cos:
                max_cos = cos
                best_entry = entry
        return best_entry, max_cos

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist model data to ``path`` using gzip-compressed pickle."""
        path = Path(path)
        if not os.path.exists(path.parent):
            os.makedirs(path.parent)
        with gzip.open(path, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self._embeddings,
                    "id_to_token": self._id_to_token,
                    "token_to_id": self._token_to_id,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CompactBert2VecModel":
        """Load a compact model from ``path``."""
        path = Path(path)
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            embeddings=data["embeddings"],
            id_to_token=data["id_to_token"],
            token_to_id=data["token_to_id"],
        )

    @classmethod
    def convert_and_save(cls, model: Bert2VecModel, path: str | Path) -> "CompactBert2VecModel":
        """Convert ``model`` to compact form and persist it to ``path``."""
        logger.info("Converting model and saving compact version to %s", path)
        compact = cls.from_bert2vec(model)
        compact.save(path)
        logger.info("Saved compact model to %s", path)
        return compact

    @classmethod
    def convert_from_path(cls, source_path: str | Path, dest_path: str | Path) -> "CompactBert2VecModel":
        """Load a ``Bert2VecModel`` from ``source_path`` and save compact version."""
        print("hello")
        logger.info("Loading Bert2VecModel from %s", source_path)
        with Bert2VecModel(source_path=source_path, in_mem=False) as model:
            return cls.convert_and_save(model, dest_path)
