import numpy as np

from src.config import config
from src.bert_2_vec_model import Bert2VecModel
from src.data_models import TokenEntry
from src.utils.models_utils import get_bert_tokenizer


def tokenize_sentence(sentence: str) -> list[str]:
    tokenizer = get_bert_tokenizer(config().bert_pretrained_name)
    return tokenizer.tokenize(sentence)


# def unite_tokens(tokens: list[str], environment bert2vec_model: Bert2VecModel) -> tuple[str, np.array]:
#     bert2vec_entries = [ver]
#


def get_bow(tokens: list, idx: int, size: int = 5):
    return tokens[idx - size if idx > size else 0 : idx] + tokens[idx + 1 : idx + size + 1]


def unite_sentence(sentence: str, bert2vec_model: Bert2VecModel, new_model: Bert2VecModel) -> str:
    tokens = tokenize_sentence(sentence)
    idx = len(tokens) - 1
    buffer: list[tuple[str, TokenEntry]] = []
    final_tokens = {}
    while idx > -1:
        token = tokens[idx]
        bow = get_bow(tokens=tokens, idx=idx)
        entry = bert2vec_model.get_entry_by_bow(token=token, bow=bow)
        buffer.append((token, entry))
        if tokens[idx].startswith("##"):
            continue

        if len(buffer) == 1:
            united_token, united_vector = unite_tokens(buffer, bert2vec_model)
        idx -= 1