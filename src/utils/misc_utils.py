import os
import shelve
from pathlib import Path

from bert2vec.model.Row import Row
from src.data_models import Embeddings, TokenEntry


def convert_rows_to_entries(source: dict[str, list[Row]]) -> Embeddings:
    return {
        token: [
            TokenEntry(bow=entry.BOW, count=entry.count, token=entry.s, vec=entry.vec, token_id=entry.tokenID)
            for entry in row
        ]
        for token, row in source.items()
    }


def convert_to_pydantic(source_path: str, dest_path: str):
    print("Loading source shelve file...")
    with shelve.open(source_path) as s:
        d = dict(s)

    print("Converting rows to entries...")
    converted = convert_rows_to_entries(source=d)

    dest_path = Path(dest_path)
    if not os.path.exists(dest_path.parent):
        os.makedirs(dest_path.parent)
    with shelve.open(str(dest_path)) as s:
        print("Updating destination shelve file with converted entries...")
        s.update(converted)
        print("Saving converted entries to destination shelve file...")
        s.sync()
