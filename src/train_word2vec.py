from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from src.utils.sqlite_sentence_streamer import SQLiteSentenceStreamer
from src.word2vec_config import Word2VecConfig, word2vec_config


class EpochLogger(CallbackAny2Vec):
    """Log progress at the end of each training epoch."""

    def __init__(self) -> None:
        self.epoch = 0

    def on_epoch_begin(self, model: Word2Vec) -> None:  # type: ignore[override]
        logging.info("Epoch %d/%d starting", self.epoch + 1, model.epochs)

    def on_epoch_end(self, model: Word2Vec) -> None:  # type: ignore[override]
        loss = model.get_latest_training_loss()
        logging.info("Epoch %d/%d finished – cumulative loss %.2f", self.epoch + 1, model.epochs, loss)
        self.epoch += 1


def train(cfg: Word2VecConfig | None = None) -> None:
    if cfg is None:
        cfg = word2vec_config()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    streamer = SQLiteSentenceStreamer(
        db_path=cfg.db_path,
        table=cfg.table,
        text_column=cfg.text_column,
        pk_column=cfg.pk_column,
        batch_size=cfg.batch_size,
        lowercase=cfg.lowercase,
        strip=cfg.strip,
    )

    corpus_size = len(streamer)
    logger.info("Detected %s sentences in %s", f"{corpus_size:,}", cfg.db_path)

    model = Word2Vec(
        vector_size=cfg.vector_size,
        window=cfg.window,
        min_count=cfg.min_count,
        sample=cfg.sample,
        negative=cfg.negative,
        workers=max(1, cfg.workers),
        sg=1 if cfg.sg else 0,
    )

    logger.info("Building vocabulary…")
    model.build_vocab(streamer, progress_per=100_000)
    logger.info("Vocabulary size: %s", f"{len(model.wv):,}")

    callbacks: list[CallbackAny2Vec] = []
    if cfg.compute_loss:
        callbacks.append(EpochLogger())

    logger.info("Starting training for %d epochs with %d worker threads", cfg.epochs, model.workers)
    model.train(
        streamer,
        total_examples=model.corpus_count,
        epochs=cfg.epochs,
        compute_loss=cfg.compute_loss,
        callbacks=callbacks,
    )

    cfg.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cfg.output_model))
    logger.info("Model saved to %s", cfg.output_model)

    if cfg.vectors_output:
        cfg.vectors_output.parent.mkdir(parents=True, exist_ok=True)
        binary = cfg.vectors_output.suffix.lower() == ".bin"
        model.wv.save_word2vec_format(str(cfg.vectors_output), binary=binary)
        logger.info("Vectors exported to %s", cfg.vectors_output)


def main(argv: list[str] | None = None) -> None:
    base_cfg = word2vec_config()

    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model from disambiguated sentences stored in SQLite."
    )
    parser.add_argument("--db-path", type=Path, default=base_cfg.db_path, help="Path to the SQLite database.")
    parser.add_argument("--table", default=base_cfg.table, help="Table containing the disambiguated sentences.")
    parser.add_argument("--text-column", default=base_cfg.text_column, help="Column with the tokenized sentences.")
    parser.add_argument(
        "--pk-column", default=base_cfg.pk_column, help="Primary key column for deterministic ordering."
    )
    parser.add_argument("--batch-size", type=int, default=base_cfg.batch_size, help="Rows to fetch per SQLite query.")
    parser.add_argument("--workers", type=int, default=base_cfg.workers, help="Number of worker threads for Word2Vec.")
    parser.add_argument("--vector-size", type=int, default=base_cfg.vector_size, help="Embedding dimensionality.")
    parser.add_argument("--window", type=int, default=base_cfg.window, help="Context window size.")
    parser.add_argument(
        "--min-count", type=int, default=base_cfg.min_count, help="Ignore tokens with frequency lower than this."
    )
    parser.add_argument(
        "--sample", type=float, default=base_cfg.sample, help="Subsampling threshold for frequent words."
    )
    parser.add_argument("--negative", type=int, default=base_cfg.negative, help="Negative sampling count.")
    parser.add_argument("--epochs", type=int, default=base_cfg.epochs, help="Number of training epochs.")
    parser.add_argument("--sg", dest="sg", action="store_true", help="Use skip-gram (default: CBOW).")
    parser.add_argument("--cbow", dest="sg", action="store_false", help="Force the CBOW architecture.")
    parser.add_argument("--lowercase", dest="lowercase", action="store_true", help="Lowercase sentences.")
    parser.add_argument("--no-lowercase", dest="lowercase", action="store_false", help="Preserve original casing.")
    parser.add_argument("--strip", dest="strip", action="store_true", help="Strip whitespace from sentences.")
    parser.add_argument("--no-strip", dest="strip", action="store_false", help="Disable stripping whitespace.")
    parser.add_argument(
        "--output-model", type=Path, default=base_cfg.output_model, help="Where to save the trained Word2Vec model."
    )
    parser.add_argument(
        "--vectors-output",
        type=Path,
        default=base_cfg.vectors_output,
        help=(
            "Optional path to export the vectors in word2vec format. Use .txt/.vec for text output or .bin for binary."
        ),
    )
    parser.add_argument("--log-level", default=base_cfg.log_level, help="Python logging level.")
    parser.add_argument("--compute-loss", dest="compute_loss", action="store_true", help="Track training loss.")
    parser.add_argument("--no-loss", dest="compute_loss", action="store_false", help="Skip loss tracking to reduce overhead.")
    parser.set_defaults(
        sg=base_cfg.sg,
        lowercase=base_cfg.lowercase,
        strip=base_cfg.strip,
        compute_loss=base_cfg.compute_loss,
    )
    args = parser.parse_args(argv)

    updated_cfg = base_cfg.model_copy(
        update={
            "db_path": args.db_path,
            "table": args.table,
            "text_column": args.text_column,
            "pk_column": args.pk_column,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "vector_size": args.vector_size,
            "window": args.window,
            "min_count": args.min_count,
            "sample": args.sample,
            "negative": args.negative,
            "epochs": args.epochs,
            "sg": args.sg,
            "lowercase": args.lowercase,
            "strip": args.strip,
            "output_model": args.output_model,
            "vectors_output": args.vectors_output,
            "log_level": args.log_level,
            "compute_loss": args.compute_loss,
        }
    )

    train(updated_cfg)


if __name__ == "__main__":
    main()
