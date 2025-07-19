import os
import shelve
from pathlib import Path
from typing import Self, Final

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from data_models import Embeddings, TokenEntry
from src.config import config
from src.data_models import TokenEntries
from src.utils.tf_utils import get_bow_idf_dict
from src.utils.models_utils import get_bert_tokenizer, get_bert_model

BERT_VECTOR_SIZE: Final[int] = 768
BERT_PRETRAINED_NAME: Final[str] = "bert-base-uncased"
ACCEPT_THRESHOLD: Final[float] = config().accecpt_threshold
RADIUS: Final[float] = config().radius


# ToDo: Rename
class Bert2VecModel:

    def __init__(
        self,
        source_path: Path | str,
        dest_path: Path | str | None = None,
        in_mem: bool = False,
    ):
        self._source_path = source_path
        self._dest_path = dest_path or source_path
        self._embeddings: Embeddings | shelve.Shelf[TokenEntries] | None = None
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
        if not path.is_file():
            raise FileNotFoundError(f"File '{self._source_path}' doesn't exist")
        return str(self._source_path)

    def load_data(self, in_mem=False) -> None:
        source_file_path = self._verify_source_table_exist()
        if not in_mem:
            self._embeddings = shelve.open(source_file_path)
        else:
            with shelve.open(source_file_path) as s:
                self._embeddings = dict(s)  # read whole

    def save_data(self) -> None:
        if isinstance(self._embeddings, dict):
            if not os.path.exists(self._dest_path.parent):
                os.makedirs(self._dest_path.parent)
            with shelve.open(self._source_path) as s:
                s.update(self._embeddings)
                s.sync()
        elif isinstance(self._embeddings, shelve.Shelf):
            self._embeddings.sync()

    def close(self):
        if isinstance(self._embeddings, shelve.Shelf):
            self._embeddings.close()

    def __getitem__(self, token: str):
        if isinstance(token, str):
            if not self._embeddings:
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
        # 1. Load the BERT tokenizer and model.
        tokenizer = get_bert_tokenizer(BERT_PRETRAINED_NAME)
        model = get_bert_model(BERT_PRETRAINED_NAME)

        # 2. Convert the sentence to tokens as PyTorch tensors.
        inputs = tokenizer(sentence, return_tensors="pt")

        # 3. Run the sentence through the BERT model without back-propagation.
        with torch.no_grad():
            outputs = model(**inputs)

        # 4. Since BERT doesn't have an output layer, the last hidden layer is our output.
        output = outputs.last_hidden_state

        # 5. We want to find the specific vector created for our token, so we first need to re-tokenize the sentence
        # to split it using BERT's splitting rules.
        sentence_tokens = tokenizer.tokenize(sentence)
        # 6. Then We convert the tokens to their IDs.
        sentence_token_ids = tokenizer.convert_tokens_to_ids(sentence_tokens)
        # 7. Retrieve the ID of our specific token.
        token_id = tokenizer.convert_tokens_to_ids(token)

        # 8. We try to extract the token index from the sentence (if it exists in the sentence).
        try:
            token_index = sentence_token_ids.index(token_id)
        except ValueError:
            raise ValueError(f"Token '{token}' not found in the sentence '{sentence}'")

        # 9. Extract the specific embedding for our token.
        bert_vector = output[0, token_index, :].numpy()

        results = []

        # 10. Now we want to find calculate how similar are our embeddings to the BERT embedding we got.
        if token in self._embeddings:
            for row in self._embeddings[token]:
                similarity = cosine_similarity([bert_vector], [row.vec])[0][0]
                results.append((row, similarity))

        # 11. Sort the results by their similarity (the most similar first).
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        return sorted_results[:max_results] if max_results else sorted_results

    def get_entries_by_bow_bm25(
        self, token: str, bow: list[str], k1: float = 1.5, max_results: int | None = None
    ) -> list[tuple[TokenEntry, float]]:
        if token not in bow:
            raise ValueError(f"Token '{token}' must be in the bag of words (BOW).")

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

    def get_entry_by_bow(self, token: str, bow: list[str]) -> TokenEntry:
        return self.get_entries_by_bow_bm25(token=token, bow=bow, max_results=1)[0][0]

    def get_entry_by_vec(self, token: str, vec: np.ndarray) -> tuple[TokenEntry | None, float]:
        entries = self._embeddings.get(token)
        closest_entry = None
        max_cos = -1
        for entry in entries:
            cos: float = cosine_similarity([entry.vec], [vec])[0][0]
            if cos > max_cos:
                max_cos = cos
                closest_entry = entry
        return closest_entry, max_cos

    def add_token(self, token: str, vec: np.ndarray, bow: list[str]) -> None:
        closest_entry, max_similarity = self.get_entry_by_vec(token=token, vec=vec)
        if closest_entry is not None:
            if max_similarity > ACCEPT_THRESHOLD:
                closest_entry.update_vec_count(new_vec=vec)
                closest_entry.update_bow(bow=bow)
            elif max_similarity < RADIUS:
                self._embeddings[token].append(TokenEntry(bow={t: 1 for t in bow}, token=token, vec=vec))
        else:
            self._embeddings[token] = [TokenEntry(bow={t: 1 for t in bow}, token=token, vec=vec)]
