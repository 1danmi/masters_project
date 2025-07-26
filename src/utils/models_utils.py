from functools import lru_cache
from typing import Final

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

from src.config import config

BERT_VECTOR_SIZE: Final[int] = 768
BERT_PRETRAINED_NAME: Final[str] = "bert-base-uncased"

# Cached tokenizer and model loaded once per process
_GLOBAL_TOKENIZER: BertTokenizer | None = None
_GLOBAL_MODEL: BertModel | None = None
_GLOBAL_DEVICE: str | None = None


def _ensure_global_model() -> tuple[BertTokenizer, BertModel]:
    """Return a tokenizer and model placed on the configured device."""
    global _GLOBAL_TOKENIZER, _GLOBAL_MODEL, _GLOBAL_DEVICE
    device = config().device
    if _GLOBAL_TOKENIZER is None or _GLOBAL_MODEL is None or _GLOBAL_DEVICE != device:
        _GLOBAL_TOKENIZER = get_bert_tokenizer(config().bert_pretrained_name)
        _GLOBAL_MODEL = get_bert_model(config().bert_pretrained_name)
        _GLOBAL_MODEL.to(device)
        _GLOBAL_MODEL.eval()
        _GLOBAL_DEVICE = device
    return _GLOBAL_TOKENIZER, _GLOBAL_MODEL


@lru_cache
def get_bert_tokenizer(pretrained_model_name: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(pretrained_model_name)


@lru_cache
def get_bert_model(pretrained_model_name: str) -> BertModel:
    return BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)


def get_bert_sentence_vectors(sentence: str) -> np.ndarray:
    """Return vectors for all tokens in the given sentence."""
    # 1. Retrieve the global tokenizer and model already loaded on the
    #    configured device (CPU by default).
    tokenizer, model = _ensure_global_model()
    device = config().device

    # 2. Tokenize the whole sentence and move the tensors to the device.
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # 3. Run the sentence through BERT without computing gradients.
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Grab the last hidden state for every token, excluding the special
    #    [CLS] and [SEP] tokens added by BERT.
    hidden_states = outputs.last_hidden_state.squeeze(0)
    return hidden_states[1:-1].cpu().numpy()


def get_bert_vec(token: str, sentence: str) -> np.ndarray:
    """Return the BERT vector of a specific token within a sentence."""
    # 1. Load the tokenizer (the model is already cached in
    #    ``get_bert_sentence_vectors``).
    tokenizer, _ = _ensure_global_model()

    # 2. Compute the embeddings for the entire sentence just once.
    sentence_vectors = get_bert_sentence_vectors(sentence)

    # 3. Tokenize again so we can locate the index of the desired token.
    tokens = tokenizer.tokenize(sentence)
    try:
        token_index = tokens.index(token)
    except ValueError:
        raise ValueError(f"Token '{token}' not found in the sentence '{sentence}'")

    # 4. Return the vector of the specified token.
    return sentence_vectors[token_index]
