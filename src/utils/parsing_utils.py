import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from src.config import config
from src.data_models import TokenEntry
from src.utils.models_utils import get_bert_tokenizer, get_bert_vec

# if TYPE_CHECKING:
#     from src.bert_2_vec_model import Bert2VecModel

def tokenize_sentence(sentence: str) -> list[str]:
    tokenizer = get_bert_tokenizer(config().bert_pretrained_name)
    return tokenizer.tokenize(sentence)


# def unite_tokens(tokens: list[str], environment bert2vec_model: Bert2VecModel) -> tuple[str, np.array]:
#     bert2vec_entries = [ver]
#


def get_bow(tokens: list, idx: int, size: int = 5):
    return tokens[idx - size if idx > size else 0 : idx] + tokens[idx + 1 : idx + size + 1]


def unite_tokens(token_list: list[tuple[str, np.ndarray]]) -> tuple[str, np.ndarray]:
    united_token = "".join(t[0].removeprefix("##") for t in token_list)
    united_vec = np.sum([t[1] for t in token_list], axis=0)
    return united_token, united_vec


def unite_sentence_tokens(sentence: str, bert2vec_model, bow_size: int = 5) -> list[TokenEntry]:
    tokens = tokenize_sentence(sentence)
    # tokens = ['his', 'behavior', 'during', 'the', 'un', '##pro', '##fe', '##ssion', '##al', 'meeting', 'up', '##set', 'both', 'clients', 'and', 'colleagues', '.']
    idx = len(tokens) - 1
    buffer: list[tuple[str, np.ndarray]] = []
    entries: list[TokenEntry] = []
    while idx > -1:
        token = tokens[idx]
        bow = get_bow(tokens=tokens, idx=idx, size=bow_size)
        entry = bert2vec_model.get_entry_by_bow(token=token, bow=bow)
        vec = entry.vec if entry else get_bert_vec(token=token, sentence=sentence)
        buffer.append((token, vec))
        if not tokens[idx].startswith("##"):
            if len(buffer) > 1:
                # We iterate the sentence from the end to start, so we need to reverse the buffer order.
                token, vec = unite_tokens(buffer[::-1])

            # The list of tokens after merging only the current tokens
            merged_tokens = tokens[:idx] + [token] + tokens[idx + len(buffer):]
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
