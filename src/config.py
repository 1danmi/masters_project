from functools import lru_cache

from pydantic_settings import BaseSettings


class ProjectConfig(BaseSettings):
    bert_pretrained_name: str = "bert-base-uncased"
    # bert2vec_path: str = "C:/Users/danie/PycharmProjects/Final Project/shared_files/shelve-unite/shelve.slv"
    bert2vec_path: str = "C:/Users/danie/PycharmProjects/Final Project/data/shelve-unite/shelve.slv"
    dest_path: str = "C:/Users/danie/PycharmProjects/Final Project/data/dest/shelve.slv"
    accept_threshold: float = 0.69
    radius: float = 0.62
    print_checkpoint_count: int = 1000
    save_checkpoint_count: int = 10000


@lru_cache
def config() -> ProjectConfig:
    return ProjectConfig()
