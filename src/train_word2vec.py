from __future__ import annotations

import logging

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


def main() -> None:
    train(word2vec_config())


if __name__ == "__main__":
    main()
