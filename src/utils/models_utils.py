from functools import lru_cache
from typing import Final

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

def get_bert_vec(token: str, sentence: str) -> np.ndarray:
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
    return output[0, token_index, :].numpy()
