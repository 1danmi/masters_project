from typing import List

from src.utils.models_utils import get_bert_tokenizer, BERT_PRETRAINED_NAME


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize *sentence* using the shared BERT tokenizer."""
    tokenizer = get_bert_tokenizer(BERT_PRETRAINED_NAME)
    return tokenizer.tokenize(sentence)


def get_bow(tokens: List[str], idx: int, size: int = 5) -> List[str]:
    """Return a window of tokens around *idx* excluding the token itself."""
    return tokens[idx - size if idx > size else 0 : idx] + tokens[idx + 1 : idx + size + 1]
