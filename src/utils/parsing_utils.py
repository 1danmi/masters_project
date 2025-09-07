import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from src.config import config
from src.data_models import TokenEntry
from src.utils.models_utils import get_bert_tokenizer, get_bert_vec

if TYPE_CHECKING:
    from src.bert_2_vec_model import Bert2VecModel
    from src.compact_bert2vec_model import CompactBert2VecModel


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


def disambiguate_sentence_tokens(
    sentence: str, bert2vec_model: "Bert2VecModel | CompactBert2VecModel", bow_size: int = 5
) -> str:
    """Return the sentence with ambiguous tokens replaced by their closest entry.

    The function tokenizes ``sentence`` using the BERT tokenizer and merges split
    word-piece tokens (``##`` prefixes) back into full words. For each merged
    token, if ``bert2vec_model`` (either a :class:`Bert2VecModel` or a
    :class:`CompactBert2VecModel`) holds more than one entry for that token, the
    entry with the closest bag-of-words (BOW) context is selected using
    ``get_entry_by_bow``. The token is then replaced with
    ``f"{token}{index}"`` where ``index`` is the position of the chosen entry in
    the model. Tokens with zero or one entry remain unchanged. Punctuation is
    kept in its original position.
    """

    tokenizer = get_bert_tokenizer(config().bert_pretrained_name)
    tokens = tokenizer.tokenize(sentence)

    # Merge word-piece tokens from end to start to rebuild full words
    idx = len(tokens) - 1
    buffer: list[str] = []
    merged_tokens: list[str] = []
    while idx > -1:
        buffer.append(tokens[idx])
        if not tokens[idx].startswith("##"):
            merged_tokens.append("".join(t.removeprefix("##") for t in buffer[::-1]))
            buffer = []
        idx -= 1
    merged_tokens = merged_tokens[::-1]

    # Resolve ambiguous tokens using the model
    resolved_tokens: list[str] = []
    for i, token in enumerate(merged_tokens):
        if hasattr(bert2vec_model, "_token_to_id") and hasattr(bert2vec_model, "_embeddings"):
            token_id = bert2vec_model._token_to_id.get(token)  # type: ignore[attr-defined]
            entries = (
                bert2vec_model._embeddings.get(token_id, [])  # type: ignore[attr-defined]
                if token_id is not None
                else []
            )
        else:
            entries = bert2vec_model[token] or []  # type: ignore[index]
        if len(entries) > 1:
            bow = get_bow(merged_tokens, i, size=bow_size)
            entry = bert2vec_model.get_entry_by_bow(token=token, bow=bow)
            if entry is not None:
                try:
                    idx_entry = next(j for j, e in enumerate(entries) if e is entry)
                    token = f"{token}{idx_entry}"
                except StopIteration:
                    pass
        resolved_tokens.append(token)

    return tokenizer.convert_tokens_to_string(resolved_tokens)
