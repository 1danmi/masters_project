import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class ProjectConfig(BaseSettings):
    bert_pretrained_name: str = "bert-base-uncased"
    bert2vec_path: str = "data/shelve-unite/shelve.slv"
    dest_path: str = "D:/temp/dest/shelve.slv"
    accept_threshold: float = 0.69
    radius: float = 0.62
    print_checkpoint_count: int = 1000
    save_checkpoint_count: int = 1000000
    workers_count: int = os.cpu_count() - 1
    chunk_size: int = 10000
    log_interval_seconds: float = 10
    results_table: str = "results"
    index_columns: str = "idx"
    input_column: str = "input"
    entries_column: str = "pickled_object"


@lru_cache
def config() -> ProjectConfig:
    return ProjectConfig()
