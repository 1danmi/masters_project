from __future__ import annotations

import pickle
import signal
import sqlite3
import itertools
from time import time
from pathlib import Path
from datetime import timedelta
from typing import Iterable, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import Dataset

from src.config import config


def chunked(it: Iterable, size: int) -> Iterable[list]:
    iterator = iter(it)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk


def format_td(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


class ParallelRunner:
    def __init__(self, db_path: Path, *, use_pickle: bool = True, input_unique: bool = True) -> None:
        self.db_path = db_path
        self.use_pickle = use_pickle
        self.input_unique = input_unique
        self.conn: sqlite3.Connection | None = None
        self.cur: sqlite3.Cursor | None = None

    # -- Context Manager -------------------------------------------------
    def __enter__(self) -> "ParallelRunner":
        self._open_db()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn:
            self.conn.commit()
            self.conn.close()

    # -- Private helpers -------------------------------------------------
    def _open_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cur = self.conn.cursor()
        column_type = "BLOB" if self.use_pickle else "TEXT"
        unique_sql = "UNIQUE" if self.input_unique else ""
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config().results_table} (
                {config().index_columns} INTEGER PRIMARY KEY,
                {config().input_column}  TEXT {unique_sql},
                {config().entries_column} {column_type} NOT NULL
            );
            """
        )
        self.conn.commit()

    def _already_done(self) -> set[int]:
        assert self.cur is not None
        self.cur.execute(f"SELECT {config().index_columns} FROM {config().results_table}")
        return {int(row[0]) for row in self.cur.fetchall()}

    # -- Public API ------------------------------------------------------
    def parallel_run(self, dataset: Dataset, func: Callable[[str], object], start_index: int = 0) -> None:
        assert self.cur is not None and self.conn is not None
        done = self._already_done()
        total_done = len(done)
        max_done = max(done) + 1 if done else 0
        if start_index < max_done:
            start_index = max_done
        print(f"Resuming: {total_done:,}/{len(dataset):,} already processed")

        # ── Graceful Ctrl‑C handling ───────────────────────────────────────────
        stop_submitting = False

        def sigint_handler(signum, frame):
            nonlocal stop_submitting
            stop_submitting = True
            print("\nReceived Ctrl+C -> will finish in flight tasks and exit...")

        signal.signal(signal.SIGINT, sigint_handler)

        # ── Process pool setup ────────────────────────────────────────────────

        total_inputs = len(dataset)
        remaining_inputs = total_inputs - start_index
        if remaining_inputs <= 0:
            print("All inputs already processed.")
            return

        print(f"Total inputs: {total_inputs:,}")
        print(f"Remaining: {remaining_inputs:,}")
        if start_index:
            print(f"Starting from dataset index {start_index:,}")

        print(f"Starting with {config().workers_count} workers")
        with ProcessPoolExecutor(max_workers=config().workers_count) as pool:
            futures = {}
            inserted_since_commit = 0
            processed = 0
            last_processed = 0
            start_time = time()
            last_log = start_time
            log_interval = config().log_interval_seconds

            dataset_iter = itertools.islice(enumerate(dataset), start_index, None)
            for chunk in chunked(dataset_iter, config().chunk_size):
                if stop_submitting:
                    break

                for idx, s in chunk:
                    text = s["text"]
                    futures[pool.submit(func, text)] = (idx, text)

                for future in as_completed(list(futures)):
                    idx, s = futures.pop(future)
                    try:
                        result_obj = future.result()
                        data = (
                            pickle.dumps(result_obj, protocol=pickle.HIGHEST_PROTOCOL)
                            if self.use_pickle
                            else result_obj
                        )
                        self.cur.execute(
                            f"INSERT OR IGNORE INTO {config().results_table} ({config().index_columns}, {config().input_column}, {config().entries_column}) VALUES (?, ?, ?)",
                            (idx, s, data),
                        )
                        inserted_since_commit += 1
                        processed += 1
                    except Exception as e:
                        print(f"Error processing {s!r}: {e!r}")

                    if inserted_since_commit >= config().save_checkpoint_count:
                        self.conn.commit()
                        inserted_since_commit = 0

                    if stop_submitting:
                        break

                    now = time()
                    if now - last_log >= log_interval:
                        elapsed = now - start_time
                        rate = processed / elapsed if elapsed else 0
                        est_total = remaining_inputs / rate if rate else 0
                        remaining = est_total - elapsed
                        print(
                            f"Progress: {processed:,}/{remaining_inputs:,} "
                            f"({100 * processed / remaining_inputs:.1f}%) – "
                            f"Elapsed: {format_td(elapsed)}, "
                            f"ETA: {format_td(remaining)}, "
                            f"Last processed: {processed-last_processed}"
                        )
                        last_log = now
                        last_processed = processed

            # In case Ctrl+C was hit
            for future in as_completed(list(futures)):
                idx, s = futures.pop(future)
                try:
                    result_obj = future.result()
                    data = pickle.dumps(result_obj, protocol=pickle.HIGHEST_PROTOCOL) if self.use_pickle else result_obj
                    self.cur.execute(
                        f"INSERT OR IGNORE INTO {config().results_table} ({config().index_columns}, {config().input_column}, {config().entries_column}) VALUES (?, ?, ?)",
                        (idx, s, data),
                    )
                    processed += 1
                except Exception as e:
                    print(f"Error processing {s!r}: {e!r}")

            self.conn.commit()
            total_time = time() - start_time

            print(f"All done. Processed {processed:,} strings in {format_td(total_time)}")


def parallel_run(
    db_path: Path,
    dataset: Dataset,
    func: Callable[[str], object],
    start_index: int = 0,
    *,
    use_pickle: bool = True,
    input_unique: bool = True,
) -> None:
    """Convenience wrapper to maintain backwards compatibility."""
    with ParallelRunner(db_path, use_pickle=use_pickle, input_unique=input_unique) as runner:
        runner.parallel_run(dataset=dataset, func=func, start_index=start_index)
