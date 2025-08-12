from functools import lru_cache
from typing import Final, Tuple, List

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

BERT_VECTOR_SIZE: Final[int] = 768
BERT_PRETRAINED_NAME: Final[str] = "bert-base-uncased"


@lru_cache
def get_bert_tokenizer(pretrained_model_name: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(pretrained_model_name)


@lru_cache
def get_bert_model(pretrained_model_name: str) -> BertModel:
    return BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)


def get_tokens_and_vectors(sentence: str) -> Tuple[List[str], List[np.ndarray]]:
    """Tokenize *sentence* and return aligned BERT vectors for all tokens.

    The sentence is encoded only once and vectors are extracted for every token,
    avoiding multiple forward passes through the BERT model.
    """

    tokenizer = get_bert_tokenizer(BERT_PRETRAINED_NAME)
    model = get_bert_model(BERT_PRETRAINED_NAME)

    # Avoid adding special tokens so that the output aligns with ``tokenize``
    tokens = tokenizer.tokenize(sentence)
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        outputs = model(**inputs)

    # ``last_hidden_state`` has shape [1, num_tokens, hidden_size]
    embeddings = outputs.last_hidden_state[0]
    return tokens, [embeddings[i].numpy() for i in range(len(tokens))]


def get_bert_vec(token: str, sentence: str) -> np.ndarray:
    """Return the BERT vector for *token* inside *sentence*.

    This helper now delegates to :func:`get_tokens_and_vectors` and is kept for
    backwards compatibility.
    """

    tokens, vectors = get_tokens_and_vectors(sentence)
    try:
        idx = tokens.index(token)
    except ValueError:
        raise ValueError(f"Token '{token}' not found in the sentence '{sentence}'")
    return vectors[idx]
