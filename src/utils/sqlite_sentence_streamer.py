"""Utilities for streaming tokenized sentences from an SQLite database."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SQLiteSentenceStreamer:
    """Stream whitespace-tokenized sentences from a SQLite table.

    The class is intentionally lightweight so it can be reused multiple times by
    libraries such as :class:`gensim.models.Word2Vec`, which iterate over the
    corpus repeatedly for vocabulary construction and training.
    """

    db_path: Path | str
    table: str
    text_column: str
    pk_column: str = "idx"
    batch_size: int = 50_000
    lowercase: bool = False
    strip: bool = True
    _row_count: int | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self._row_count is None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {self.table}")
                self._row_count = int(cursor.fetchone()[0])
        return self._row_count

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[list[str]]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF;")
            conn.execute("PRAGMA synchronous=OFF;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            cursor = conn.cursor()
            last_pk = -1
            sql = (
                f"SELECT {self.pk_column}, {self.text_column} "
                f"FROM {self.table} "
                f"WHERE {self.pk_column} > ? "
                f"ORDER BY {self.pk_column} "
                f"LIMIT ?"
            )

            while True:
                rows = cursor.execute(sql, (last_pk, self.batch_size)).fetchall()
                if not rows:
                    break

                for pk, sentence in rows:
                    last_pk = pk
                    if sentence is None:
                        continue

                    if self.strip:
                        sentence = sentence.strip()
                    if not sentence:
                        continue

                    if self.lowercase:
                        sentence = sentence.lower()

                    tokens = sentence.split()
                    if tokens:
                        yield tokens
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def head(self, n: int) -> list[list[str]]:
        """Return the first ``n`` tokenized sentences for inspection."""

        out: list[list[str]] = []
        for idx, sentence in enumerate(self):
            if idx >= n:
                break
            out.append(sentence)
        return out

    # ------------------------------------------------------------------
    def take(self, n: int) -> Iterable[list[str]]:
        """Yield at most ``n`` sentences (useful for sampling)."""

        for idx, sentence in enumerate(self):
            if idx >= n:
                return
            yield sentence


__all__ = ["SQLiteSentenceStreamer"]
