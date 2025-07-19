from functools import lru_cache

from transformers import BertTokenizer, BertModel


@lru_cache
def get_bert_tokenizer(pretrained_model_name: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(pretrained_model_name)


@lru_cache
def get_bert_model(pretrained_model_name: str) -> BertModel:
    return BertModel.from_pretrained(pretrained_model_name, output_hidden_states=True)

