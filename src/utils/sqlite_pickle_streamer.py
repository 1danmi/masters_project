from __future__ import annotations

import json
import pickle
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import config


class SQLitePickleStreamer[T]:
    """
    Stream rows from an SQLite table in pk-ordered chunks, unpickle a BLOB column,
    call a user callback sequentially, checkpoint progress, and—optionally—
    delete rows that were processed successfully.
    """

    __slots__ = (
        "db_path",
        "table",
        "pk_col",
        "blob_col",
        "chunk_size",
        "state_path",
        "stop_flag_path",
        "log_every_sec",
        "unpickle_fn",
        "extra_cols",
        "connection_kwargs",
        "delete_processed",
        "_state",
        "_conn",
        "_rows_to_process",
    )

    # ---------- constructor ----------
    def __init__(
        self,
        db_path: str | Path,
        table: str,
        pk_col: str,
        blob_col: str,
        *,
        chunk_size: int = 1000,
        state_path: str | Path = "progress_state.json",
        stop_flag_path: str | Path = "STOP",
        log_every_sec: float = 5.0,
        unpickle_fn: Callable[[bytes], T] = pickle.loads,
        extra_cols: list[str] | None = None,
        connection_kwargs: dict[str, Any] | None = None,
        delete_processed: bool = False,  # <-- NEW FLAG
    ) -> None:
        # user-supplied parameters
        self.db_path = str(db_path)
        self.table = table
        self.pk_col = pk_col
        self.blob_col = blob_col
        self.chunk_size = chunk_size
        self.state_path = Path(state_path)
        self.stop_flag_path = Path(stop_flag_path)
        self.log_every_sec = log_every_sec
        self.unpickle_fn = unpickle_fn
        self.extra_cols = extra_cols or []
        self.connection_kwargs = connection_kwargs or {}
        self.delete_processed = delete_processed  # <-- store

        # internal state
        self._state: dict[str, int] = {"last_pk": 0, "processed": 0}
        self._conn: sqlite3.Connection | None = None
        self._rows_to_process: int | None = None

    def _open(self) -> None:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, **self.connection_kwargs)
            self._conn.row_factory = sqlite3.Row

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _fetch_chunk(self, last_pk: int) -> list[sqlite3.Row]:
        assert self._conn is not None
        cols = [self.pk_col, self.blob_col, *self.extra_cols]
        sql = f"""
            SELECT {', '.join(cols)}
            FROM {self.table}
            WHERE {self.pk_col} > ?
            ORDER BY {self.pk_col}
            LIMIT ?
        """
        return self._conn.execute(sql, (last_pk, self.chunk_size)).fetchall()

    def _count_total_rows(self) -> int:
        assert self._conn is not None
        return self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

    def _delete_rows(self, pks: list[int]) -> None:
        """Delete processed rows in one statement and commit."""
        assert self._conn is not None
        placeholders = ",".join("?" * len(pks))
        sql = f"DELETE FROM {self.table} WHERE {self.pk_col} IN ({placeholders})"
        self._conn.execute(sql, pks)
        self._conn.commit()
        print(f"Deleted {len(pks)} processed rows.")

    def _load_state(self) -> None:
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def _checkpoint(self, last_pk: int, processed: int) -> None:
        self._state.update(last_pk=last_pk, processed=processed)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state), encoding="utf-8")
        try:
            tmp.replace(self.state_path)
        except Exception:
            time.sleep(1)
            tmp.replace(self.state_path)

    @staticmethod
    def _eta(start_time: float, processed: int, total: int) -> float | None:
        if processed == 0:
            return None
        rate = processed / (time.perf_counter() - start_time)
        return (total - processed) / rate if rate > 0 else None

    @staticmethod
    def _fmt_seconds(sec: float | None) -> str:
        if sec is None:
            return "?"
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def run(self, callback: Callable[[T, int, dict[str, Any]], None]) -> None:
        """
        Start / resume processing.
        `callback(obj: T, pk: int, extras: dict[str, Any])` is called sequentially.
        """
        self._load_state()
        self._open()

        last_pk = self._state["last_pk"]
        starting_pk = last_pk
        processed = self._state["processed"]

        total_rows = self._count_total_rows()
        self._rows_to_process = total_rows if self.delete_processed else total_rows - processed

        print(
            f"Starting from {self.pk_col} > {last_pk} (processed {processed} / {total_rows}).  delete_processed={self.delete_processed}",
        )

        start_time = time.perf_counter()
        last_log = start_time

        try:
            while True:
                if self.stop_flag_path.exists():
                    print("STOP flag detected. Saving state and exiting.")
                    break

                rows = self._fetch_chunk(last_pk)
                if not rows:
                    print(
                        f"Done. Processed {processed} / {self._rows_to_process} rows.", processed, self._rows_to_process
                    )
                    break

                _to_delete: list[int] = []  # PKs to remove after chunk

                for row in rows:
                    pk = row[self.pk_col]
                    obj: T = self.unpickle_fn(row[self.blob_col])
                    extras = {c: row[c] for c in self.extra_cols} if self.extra_cols else {}

                    callback(obj, pk, extras)  # user code

                    processed += 1
                    last_pk = pk
                    if self.delete_processed:
                        _to_delete.append(pk)

                    now = time.perf_counter()
                    if now - last_log >= self.log_every_sec:
                        self._checkpoint(last_pk, processed)
                        eta = self._eta(start_time, processed - starting_pk, self._rows_to_process)
                        print(
                            f"Processed {processed-starting_pk}/{self._rows_to_process} ({(processed - starting_pk) * 100 / self._rows_to_process:.2f}%). ETA {self._fmt_seconds(eta) if eta is not None else '?'}",
                        )
                        last_log = now

                # --- post-chunk cleanup ---
                if _to_delete:
                    self._delete_rows(_to_delete)

                # if (processed - starting_pk) % config().save_checkpoint_count == 0:
                #     print("Vacuuming...")
                #     self._conn.execute("VACUUM;")

                self._checkpoint(last_pk, processed)

        except KeyboardInterrupt:
            print("KeyboardInterrupt. Saving state and exiting...")
            self._checkpoint(last_pk, processed)
            raise
        except Exception:  # noqa: BLE001  (intentional for checkpoint)
            print("Crash! Saving state and re-raising.")
            self._checkpoint(last_pk, processed)
            raise
        finally:
            self._close()
