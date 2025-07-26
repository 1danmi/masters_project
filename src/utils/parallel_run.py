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


SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {config().results_table} (
    {config().index_columns} INTEGER PRIMARY KEY,
    {config().input_column}  TEXT UNIQUE,
    {config().entries_column} BLOB NOT NULL
);
"""


def init_db(path: Path) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")  # Better concurrency / crash resiliency
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    conn.commit()
    return conn, cur


def already_done(cur: sqlite3.Cursor) -> set[str]:
    cur.execute(f"SELECT {config().input_column} FROM {config().results_table}")
    return {row[0] for row in cur.fetchall()}


def stream_inputs(file_path: Path, skip: set[str]) -> Iterable[str]:
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s and s not in skip:
                yield s


def chunked(it: Iterable[str], size: int) -> Iterable[list[str]]:
    iterator = iter(it)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk


def format_td(seconds: float) -> str:
    return str(timedelta(seconds=seconds))


def parallel_run(db_path: Path, dataset: Dataset, func: Callable[[str], object]):
    conn, cur = init_db(db_path)
    done = already_done(cur)
    total_done = len(done)
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
    remaining_inputs = total_inputs - total_done
    if remaining_inputs <= 0:
        print("All inputs already processed.")
        return

    print(f"Total inputs: {total_inputs:,}")
    print(f"Remaining: {remaining_inputs:,}")

    with ProcessPoolExecutor(max_workers=config().workers_count) as pool:
        futures = {}
        inserted_since_commit = 0
        processed = 0

        start_time = time()
        last_log = start_time
        log_interval = config().log_interval_seconds

        for chunk in chunked(dataset, config().chunk_size):
            if stop_submitting:
                break

            for s in chunk:
                s = s["text"]
                futures[pool.submit(func, s)] = s

            for future in as_completed(list(futures)):
                s = futures.pop(future)
                try:
                    result_obj = future.result()
                    pickled = pickle.dumps(result_obj, protocol=pickle.HIGHEST_PROTOCOL)
                    cur.execute(
                        f"INSERT OR IGNORE INTO {config().results_table} ({config().input_column}, {config().entries_column}) VALUES (?, ?)",
                        (s, pickled),
                    )
                    inserted_since_commit += 1
                    processed += 1
                except Exception as e:
                    print(f"Error processing {s!r}: {e!r}")

                if inserted_since_commit >= config().save_checkpoint_count:
                    conn.commit()
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
                        f"ETA: {format_td(remaining)}"
                    )
                    last_log = now

        # In case Ctrl+C was hit
        for future in as_completed(list(futures)):
            s = futures.pop(future)
            try:
                result_obj = future.result()
                pickled = pickle.dumps(result_obj, protocol=pickle.HIGHEST_PROTOCOL)
                cur.execute(
                    f"INSERT OR IGNORE INTO {config().results_table} ({config().input_column}, {config().entries_column}) VALUES (?, ?)",
                    (s, pickled),
                )
                processed += 1
            except Exception as e:
                print(f"Error processing {s!r}: {e!r}")

        conn.commit()
        conn.close()
        total_time = time() - start_time

    print(f"All done. Processed {processed:,} strings in {format_td(total_time)}")
