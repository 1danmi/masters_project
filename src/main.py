import os
import shelve
from datetime import timedelta
from pathlib import Path
import logging
from time import time

import numpy as np
from datasets import load_dataset, Dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.fx.experimental.unification.multipledispatch.dispatcher import source

from bert2vec.model.Row import Row
from src.bert_2_vec_model import Bert2VecModel
from src.config import config
from src.data_models import TokenEntry
from src.utils.parsing_utils import tokenize_sentence, unite_sentence_tokens
from src.utils.misc_utils import convert_rows_to_entries, convert_to_pydantic

# logger = logging.getLogger()
logging.basicConfig()

def get_examples_with_word(
    word: str, book_corpus: Dataset, num_of_examples: int = 1, allow_less: bool = False
) -> dict[int, str]:
    examples = {}
    for idx, example in enumerate(book_corpus):
        if text := example.get("text", ""):
            if word in text.split():
                print(f"Found '{word}' in the book corpus at index {idx}.")
                examples[idx] = text
                if len(examples) == num_of_examples:
                    return examples
    if allow_less:
        return examples
    raise ValueError(f"Word '{word}' not found enough times in the book corpus.")

def create_new_model(corpus: list[dict[str, str]]):
    sentence_count = len(corpus)
    with Bert2VecModel(source_path=config().bert2vec_path, in_mem=False) as bert2vec_model:
        with Bert2VecModel(source_path=config().dest_path, in_mem=False, new_model=True) as dest_model:
            last_save_checkpoint = 0
            print(f"Starting...")
            start_time = epoch_time = time()
            for idx, sentence in enumerate(corpus):
                sentence = sentence["text"]
                sentence_entries = unite_sentence_tokens(sentence=sentence, bert2vec_model=bert2vec_model)
                for entry in sentence_entries:
                    dest_model.add_entry(entry)
                if idx%config().print_checkpoint_count == 0:
                    end_time = time()
                    run_time = end_time - epoch_time
                    total_time = end_time - start_time
                    average_time = total_time/(idx+1)
                    time_remaining = (sentence_count - idx + 1)*average_time
                    print(f"""Finished {idx+1:,}/{sentence_count:,} sentences in {run_time:.4f} seconds\n"""
                          f"""Total time: {timedelta(seconds=total_time)}\n"""
                          f"""Average time: {average_time} seconds/sentence,\n"""
                          f"""Time remaining: {timedelta(seconds=time_remaining)}\n"""
                          f"""Last saved checkpoint: {last_save_checkpoint:,} sentences""")
                    epoch_time = end_time
                if idx%config().save_checkpoint_count == 0:
                    last_save_checkpoint = idx
                    dest_model.save_data()
    print(f"Done {sentence_count} sentences")


def main():
    # source_path = config().bert2vec_path
    # source_path = "C:/Users/danie/PycharmProjects/Final Project/shared_files/shelve-unite/shelve.slv"
    # dest_path = "C:/Users/danie/PycharmProjects/Final Project/data/shelve-unite/shelve.slv"
    # convert_to_pydantic(source_path=source_path, dest_path=dest_path)
    WORD = "light"
    NUM_OF_EXAMPLES = 100
    print("Loading dataset...")
    book_corpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    print("Done loading dataset, starting building model...")
    create_new_model(corpus=book_corpus)
    # examples = list(get_examples_with_word(word=WORD, book_corpus=book_corpus, num_of_examples=NUM_OF_EXAMPLES).values())


            # example = "This bank is unprofessional at all today."
            # examples = [
            #     # "His behavior during the unprofessional meeting upset both clients and colleagues.",
            #     # "He went to the bank to get a load for his house",
            #     # "They really wanted to cross to the other bank of the wide river",
            #     # "This book was very long but interesting",
            #     # "He used this website to book an appointment for his barber."
            #     "Please turn on the light, it’s too dark in here.",
            #     "This suitcase is very light, I can carry it with one hand."
            # ]
            # entries = [bert2vec_model.get_entry_by_sentence(token=WORD, sentence=example) for example in examples]
            # similarities = np.zeros((NUM_OF_EXAMPLES, NUM_OF_EXAMPLES))
            # for i, entry1 in enumerate(entries):
            #     for j, entry2 in enumerate(entries):
            #         similarities[i][j] = cosine_similarity([entry1.vec], [entry2.vec])

            # data = {}
            # dest_model.add_entry(entry1)
            # dest_model.add_entry(entry2)


            # x = 3

        # # 1. Tokenize the example sentence
        # tokens = tokenize_sentence(example)
        # entry = bert2vec_model.get_entry_by_bow(token="bank", bow=tokens)
        # result = bert2vec_model.get_entry_by_vec(token="bank", vec=entry.vec)
        # print(tokens)


if __name__ == "__main__":
    main()
