from collections import Counter
from typing import List, Tuple

import numpy as np

from src.data_models import TokenEntry
from src.utils.model_worker import get_model
from src.utils.models_utils import get_tokens_and_vectors
from src.utils.tokenization_utils import get_bow


def unite_tokens(token_list: List[Tuple[str, np.ndarray]]) -> Tuple[str, np.ndarray]:
    united_token = "".join(t[0].removeprefix("##") for t in token_list)
    united_vec = np.sum([t[1] for t in token_list], axis=0)
    return united_token, united_vec


def unite_sentence_tokens(sentence: str, bow_size: int = 5) -> List[TokenEntry]:
    bert2vec_model = get_model()
    tokens, token_vecs = get_tokens_and_vectors(sentence)
    idx = len(tokens) - 1
    buffer: List[Tuple[str, np.ndarray]] = []
    entries: List[TokenEntry] = []
    while idx > -1:
        token = tokens[idx]
        bow = get_bow(tokens=tokens, idx=idx, size=bow_size)
        entry = bert2vec_model.get_entry_by_bow(token=token, bow=bow)
        vec = entry.vec if entry else token_vecs[idx]
        buffer.append((token, vec))
        if not tokens[idx].startswith("##"):
            if len(buffer) > 1:
                # We iterate the sentence from the end to start, so we need to reverse the buffer order.
                token, vec = unite_tokens(buffer[::-1])

            # The list of tokens after merging only the current tokens
            merged_tokens = tokens[:idx] + [token] + tokens[idx + len(buffer) :]
            bow_b2v = get_bow(tokens=merged_tokens, idx=idx, size=bow_size)

            entry = TokenEntry(bow_b2v=dict(Counter(bow_b2v)), token=token, vec=vec)

            entries.append(entry)
            buffer = []
        idx -= 1

    # Create the BOW after all the words where merged
    merged_tokens = [entry.token for entry in entries[::-1]]
    for idx, entry in enumerate(entries):
        entry.bow = dict(Counter(get_bow(tokens=merged_tokens, idx=idx, size=bow_size)))

    # We iterate the sentence from the end to start, so we need to reverse the order of the final results.
    return entries[::-1]
