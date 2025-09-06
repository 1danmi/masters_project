import logging
from pathlib import Path
from functools import partial
from typing import Any

from datasets import load_dataset, Dataset
from tqdm import tqdm

from src.config import config
from src.bert_2_vec_model import Bert2VecModel
from src.data_models import TokenEntry
from src.utils.parallel_run import parallel_run
from src.utils.parsing_utils import unite_sentence_tokens, disambiguate_sentence_tokens
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


def create_entries_db(dataset: Dataset, start_index: int = 0):
    db_path = Path("data/temp/data.db")
    with Bert2VecModel(source_path=config().bert2vec_path, in_mem=False) as bert2vec_model:
        func = partial(unite_sentence_tokens, bert2vec_model=bert2vec_model)
        parallel_run(db_path=db_path, dataset=dataset, func=func, start_index=start_index)


def disambiguate_dataset(dataset: Dataset, model_path: str, start_index: int = 0) -> None:
    """Run ``disambiguate_sentence_tokens`` on ``dataset`` in parallel.

    Results are stored as plain text in a SQLite database whose path is
    configured via ``config().disambiguated_db_path``. Each row is keyed by the
    original dataset index so the output can be matched back to its source
    sentence. The underlying runner supports resuming by reading existing
    entries from the database and can be interrupted with Ctrl-C.
    """

    db_path = Path(config().disambiguated_db_path)
    with Bert2VecModel(source_path=model_path, in_mem=False) as bert2vec_model:
        func = partial(disambiguate_sentence_tokens, bert2vec_model=bert2vec_model)
        parallel_run(
            db_path=db_path,
            dataset=dataset,
            func=func,
            start_index=start_index,
            use_pickle=False,
            input_unique=False,
        )


counter = 1


def update_model():
    db_path = Path("data/temp/data.db")
    with Bert2VecModel(source_path=config().dest_path, in_mem=False) as dest_model:

        def my_cb(entries: list[TokenEntry], pk: int, extras: dict[str, Any]) -> None:
            global counter
            for entry in entries:
                dest_model.add_entry(entry)
                counter += 1
            if counter % config().save_checkpoint_count == 0:
                dest_model.save_data()
                print(f"Saving model, currently holding {len(dest_model._embeddings)}...")

        streamer = SQLitePickleStreamer[list[TokenEntry]](
            db_path=db_path,
            table=config().results_table,
            pk_col=config().index_columns,
            blob_col=config().entries_column,
            delete_processed=False,
            extra_cols=[config().input_column],
        )

        streamer.run(my_cb)


def replace_tokens(model: Bert2VecModel, sentence: str):
    # sentence = "The bank is very unprofessional today"
    # sentence = "The gross river bank was really far away."
    print(disambiguate_sentence_tokens(sentence=sentence, bert2vec_model=model))


def main():
    # print("Loading dataset...")
    dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    # print("Done loading dataset, starting building model...")
    # create_entries_db(dataset=dataset, start_index=33164770)
    # update_model()
    with Bert2VecModel(source_path=config().dest_path, in_mem=False) as model:
        count = 0
        while count < 10:
            for text in dataset:
                if " book " in text["text"]:
                    replace_tokens(model=model, sentence=text["text"])
                    count += 1


if __name__ == "__main__":
    main()
