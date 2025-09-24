from __future__ import annotations

import argparse
import logging
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from src.config import config
from src.utils.sqlite_sentence_streamer import SQLiteSentenceStreamer


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Word2Vec model from disambiguated sentences stored in SQLite."
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path(config().disambiguated_db_path), help="Path to the SQLite database."
    )
    parser.add_argument("--table", default=config().results_table, help="Table containing the disambiguated sentences.")
    parser.add_argument(
        "--text-column", default=config().disambiguated_column, help="Column with the tokenized sentences."
    )
    parser.add_argument(
        "--pk-column", default=config().index_columns, help="Primary key column for deterministic ordering."
    )
    parser.add_argument("--batch-size", type=int, default=50_000, help="How many rows to fetch per SQLite query.")
    parser.add_argument(
        "--workers", type=int, default=config().workers_count, help="Number of worker threads for Word2Vec."
    )
    parser.add_argument("--vector-size", type=int, default=100, help="Embedding dimensionality (default: 100).")
    parser.add_argument("--window", type=int, default=5, help="Context window size (default: 5).")
    parser.add_argument(
        "--min-count", type=int, default=5, help="Ignore tokens with total frequency lower than this (default: 5)."
    )
    parser.add_argument(
        "--sample", type=float, default=1e-3, help="Subsampling threshold for frequent words (default: 1e-3)."
    )
    parser.add_argument("--negative", type=int, default=5, help="Negative sampling count (default: 5).")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--sg", action="store_true", help="Use skip-gram instead of CBOW (default: CBOW).")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase sentences before tokenization.")
    parser.add_argument(
        "--no-strip", dest="strip", action="store_false", help="Disable stripping whitespace from sentences."
    )
    parser.add_argument("--output-model", type=Path, required=True, help="Where to save the trained Word2Vec model.")
    parser.add_argument(
        "--vectors-output",
        type=Path,
        help=(
            "Optional path to also export the vectors in word2vec format. Use a .txt or .vec extension for text output, "
            "or .bin for binary."
        ),
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    parser.add_argument(
        "--no-loss", dest="compute_loss", action="store_false", help="Skip loss tracking to reduce overhead."
    )
    parser.set_defaults(strip=True, compute_loss=True)
    return parser


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    streamer = SQLiteSentenceStreamer(
        db_path=args.db_path,
        table=args.table,
        text_column=args.text_column,
        pk_column=args.pk_column,
        batch_size=args.batch_size,
        lowercase=args.lowercase,
        strip=args.strip,
    )

    corpus_size = len(streamer)
    logger.info("Detected %s sentences in %s", f"{corpus_size:,}", args.db_path)

    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        sample=args.sample,
        negative=args.negative,
        workers=max(1, args.workers),
        sg=1 if args.sg else 0,
    )

    logger.info("Building vocabulary…")
    model.build_vocab(streamer, progress_per=100_000)
    logger.info("Vocabulary size: %s", f"{len(model.wv):,}")

    callbacks: list[CallbackAny2Vec] = []
    if args.compute_loss:
        callbacks.append(EpochLogger())

    logger.info("Starting training for %d epochs with %d worker threads", args.epochs, model.workers)
    model.train(
        streamer,
        total_examples=model.corpus_count,
        epochs=args.epochs,
        compute_loss=args.compute_loss,
        callbacks=callbacks,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output_model))
    logger.info("Model saved to %s", args.output_model)

    if args.vectors_output:
        args.vectors_output.parent.mkdir(parents=True, exist_ok=True)
        binary = args.vectors_output.suffix.lower() == ".bin"
        model.wv.save_word2vec_format(str(args.vectors_output), binary=binary)
        logger.info("Vectors exported to %s", args.vectors_output)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
