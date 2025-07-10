import shelve
from typing import Self
from pathlib import Path

from models import Embeddings


# ToDo: Rename
class ModelManager:
    def __init__(
        self,
        source_path: Path | str,
        dest_path: Path | str | None = None,
        lazy_load: bool = False,
        in_mem: bool = False,
    ):
        self._source_path = self._validate_path(source_path)
        self._dest_path = self._validate_path(dest_path) if dest_path else source_path
        self._embeddings: Embeddings | shelve.Shelf | None = None
        if not lazy_load:
            self.load_data(in_mem=in_mem)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _validate_path(path: Path | str) -> Path:
        path = Path(path)
        if path.suffix != ".dat":
            path = path.with_suffix(".dat")
        return path


    def _verify_source_file_exist(self) -> str:
        if not self._source_path.is_file():
            raise FileNotFoundError(f"File '{self._source_path}' doesn't exist")
        return str(self._source_path)

    def load_data(self, in_mem=False):
        source_file_path = self._verify_source_file_exist()
        if not in_mem:  # No need to close shelve, just open from HD as dict
            self._embeddings = shelve.open(source_file_path)
        else:
            with shelve.open(source_file_path) as s:
                self._embeddings = dict(s)  # read whole

    def close(self):
        if isinstance(self._embeddings, shelve.Shelf):
            self._embeddings.close()