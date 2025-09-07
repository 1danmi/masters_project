import logging
from pathlib import Path
from typing import Any

from datasets import load_dataset, Dataset

from src import CompactBert2VecModel
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


_worker_model: Bert2VecModel | CompactBert2VecModel | None = None


def _init_worker(model_path: str, compact: bool = False) -> None:
    """Initialize ``_worker_model`` for worker processes."""
    global _worker_model
    if compact:
        _worker_model = CompactBert2VecModel.load(model_path)
    else:
        _worker_model = Bert2VecModel(source_path=model_path, in_mem=False)


def _unite_worker(sentence: str):
    assert _worker_model is not None
    return unite_sentence_tokens(sentence=sentence, bert2vec_model=_worker_model)


def _disambiguate_worker(sentence: str) -> str:
    assert _worker_model is not None
    return disambiguate_sentence_tokens(sentence=sentence, bert2vec_model=_worker_model)


def create_entries_db(dataset: Dataset, start_index: int = 0) -> None:
    db_path = Path("data/temp/data.db")
    parallel_run(
        db_path=db_path,
        dataset=dataset,
        func=_unite_worker,
        start_index=start_index,
        initializer=_init_worker,
        initargs=(config().bert2vec_path, False),
    )


def disambiguate_dataset(dataset: Dataset, model_path: str, start_index: int = 0, compact: bool = False) -> None:
    """Run ``disambiguate_sentence_tokens`` on ``dataset`` in parallel.

    Results are stored as plain text in a SQLite database whose path is
    configured via ``config().disambiguated_db_path``. Each row is keyed by the
    original dataset index so the output can be matched back to its source
    sentence. The underlying runner supports resuming by reading existing
    entries from the database and can be interrupted with Ctrl-C.
    """

    db_path = Path(config().disambiguated_db_path)
    parallel_run(
        db_path=db_path,
        dataset=dataset,
        func=_disambiguate_worker,
        start_index=start_index,
        use_pickle=False,
        input_unique=False,
        initializer=_init_worker,
        initargs=(model_path, compact),
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


def replace_tokens(model_path):
    # sentence = "The bank is very unprofessional today"
    model = CompactBert2VecModel.load(model_path)
    sentence = "The gross river bank was really far away."
    print(disambiguate_sentence_tokens(sentence=sentence, bert2vec_model=model))


def main():
    import logging, sys

    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # <— wipes existing ROOT handlers so this actually takes effect
    )
    print("Loading dataset...")
    dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    print("Done loading dataset, starting building model...")
    # create_entries_db(dataset=dataset, start_index=33164770)
    # update_model()
    # replace_tokens(model_path=config().compact_dest_path)
    disambiguate_dataset(dataset=dataset, model_path=config().compact_dest_path, compact=True)
    # CompactBert2VecModel.convert_from_path(source_path=config().dest_path, dest_path=config().compact_dest_path)


if __name__ == "__main__":
    main()
