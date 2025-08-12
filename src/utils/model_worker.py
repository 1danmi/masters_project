"""Utilities for sharing a single :class:`Bert2VecModel` per worker process.

Each worker process loads the model once via :func:`init_worker` and later
retrieves it using :func:`get_model`.  This avoids repeatedly pickling the
model for every submitted task.
"""

from __future__ import annotations

import atexit
from typing import Optional

from src.bert_2_vec_model import Bert2VecModel
from src.config import config


_model: Optional[Bert2VecModel] = None


def _close_model() -> None:
    global _model
    if _model is not None:
        _model.close()
        _model = None


def init_worker() -> None:
    """Load the :class:`Bert2VecModel` once per worker process."""

    global _model
    if _model is None:
        _model = Bert2VecModel(source_path=config().bert2vec_path, in_mem=False)
        atexit.register(_close_model)


def get_model() -> Bert2VecModel:
    """Return the worker's global :class:`Bert2VecModel` instance."""

    if _model is None:
        raise RuntimeError("Model not initialised – did you forget to call init_worker?")
    return _model
