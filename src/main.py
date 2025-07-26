import logging
from pathlib import Path
from functools import partial

from datasets import load_dataset, Dataset

from src.config import config
from src.bert_2_vec_model import Bert2VecModel
from src.utils.parallel_run import parallel_run
from src.utils.parsing_utils import unite_sentence_tokens

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


def main():
    print("Loading dataset...")
    dataset = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)["train"]
    print("Done loading dataset, starting building model...")
    create_entries_db(dataset=dataset)


if __name__ == "__main__":
    main()
