import logging
from pathlib import Path
from functools import partial
from typing import Any

from datasets import load_dataset, Dataset

from src.config import config
from src.bert_2_vec_model import Bert2VecModel
from src.data_models import TokenEntry
from src.utils.parallel_run import parallel_run
from src.utils.parsing_utils import unite_sentence_tokens
from src.utils.sqlite_pickle_streamer import SQLitePickleStreamer

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


def create_entries_db(dataset: Dataset):
    db_path = Path("data/temp/data.db")
    with Bert2VecModel(source_path=config().bert2vec_path, in_mem=False) as bert2vec_model:
        func = partial(unite_sentence_tokens, bert2vec_model=bert2vec_model)
        parallel_run(db_path=db_path, dataset=dataset, func=func)

counter = 0

def update_model():
    db_path = Path("data/temp/data.db")
    with Bert2VecModel(source_path=config().dest_path, in_mem=True) as dest_model:
        def my_cb(entries: list[TokenEntry], pk: int, extras: dict[str, Any]) -> None:
            global counter
            counter += 1
            for entry in entries:
                dest_model.add_entry(entry)
            if counter % config().save_checkpoint_count == 0:
                print(f"Saving model...")
                dest_model.save_data()


        streamer = SQLitePickleStreamer[list[TokenEntry]](
            db_path=db_path,
            table=config().results_table,
            pk_col=config().index_columns,
            blob_col=config().entries_column,
            delete_processed=True,
            extra_cols=[config().input_column],
        )

        streamer.run(my_cb)



def main():
    # print("Loading dataset...")
    # dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    # print("Done loading dataset, starting building model...")
    # create_entries_db(dataset=dataset)
    update_model()


if __name__ == "__main__":
    main()
