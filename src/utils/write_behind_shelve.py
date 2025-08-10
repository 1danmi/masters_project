import threading
import shelve
from collections.abc import MutableMapping
from typing import Any, Iterator


class WriteBehindShelf(MutableMapping):
    """A shelf-like key-value store with a write-behind cache.

    Values written via ``__setitem__`` are stored in an in-memory cache and
    flushed to disk either when the cache reaches ``max_cache_size`` entries or
    when ``flush_interval`` seconds have passed (if provided).
    """

    def __init__(self, path: str, max_cache_size: int = 1000, flush_interval: float | None = None):
        self._shelf = shelve.open(path, writeback=False)
        self._cache: dict[str, Any] = {}
        self._max_cache_size = max_cache_size
        self._flush_interval = flush_interval
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        if flush_interval:
            self._start_timer()

    # Internal helper methods -------------------------------------------------
    def _start_timer(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._periodic_flush)
        self._timer.daemon = True
        self._timer.start()

    def _periodic_flush(self) -> None:
        self.flush()
        # restart the timer for the next interval
        if self._flush_interval:
            self._start_timer()

    # MutableMapping interface ------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            return self._shelf[key]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache.get(key, default)
            return self._shelf.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            if len(self._cache) >= self._max_cache_size:
                self.flush()

    def __delitem__(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)
            del self._shelf[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            keys = set(self._shelf.keys()) | set(self._cache.keys())
        return iter(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(set(self._shelf.keys()) | set(self._cache.keys()))

    # Public API --------------------------------------------------------------
    def flush(self) -> None:
        """Flush cached values to disk immediately."""
        print(f"Flushing...")
        with self._lock:
            if self._cache:
                self._shelf.update(self._cache)
                self._shelf.sync()
                self._cache.clear()

    # Alias used by existing code
    sync = flush

    def close(self) -> None:
        if self._timer:
            self._timer.cancel()
        self.flush()
        self._shelf.close()
