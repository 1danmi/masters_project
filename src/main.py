import os
import shelve
from pathlib import Path

from datasets import load_dataset, Dataset
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from bert2vec.model.Row import Row
from src.bert_2_vec_model import Bert2VecModel
from src.config import config
from src.data_models import TokenEntry
from src.utils.parsing_utils import tokenize_sentence
from src.utils.misc_utils import convert_rows_to_entries


def get_examples_with_word(word: str, book_corpus: Dataset, num_of_examples: int = 1, allow_less: bool = False) -> dict[int, str]:
    examples = {}
    for idx, example in enumerate(book_corpus):
        if text:=example.get("text", ""):
            if word in text:
                print(f"Found '{word}' in the book corpus at index {idx}.")
                examples[idx] = text
                if len(examples) == num_of_examples:
                    return examples
    if allow_less:
        return examples
    raise ValueError(f"Word '{word}' not found enough times in the book corpus.")





def main():
    # source_path = config().bert2vec_path
    # dest_path = "C:/Users/danie/PycharmProjects/Final Project/data/shelve-unite/shelve.slv"
    # convert_to_pydantic(source_path=source_path, dest_path=dest_path)
    # book_corpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    # example = book_corpus[7068]["text"]
    with Bert2VecModel(source_path=config().bert2vec_path, in_mem=False) as bert2vec_model:
        example = "This bank is unprofessional at all today."


        # 1. Tokenize the example sentence
        tokens = tokenize_sentence(example)
        entry = bert2vec_model.get_entry_by_bow(token="bank", bow=tokens)
        result = bert2vec_model.get_entry_by_vec(token="bank", vec=entry.vec)
        print(tokens)






if __name__ == "__main__":
    main()
