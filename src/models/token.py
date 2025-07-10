import numpy as np
from pydantic import BaseModel, RootModel


class TokenEntry(BaseModel):
    bow: dict[str, int]
    count: int
    s: str
    token_id: int
    vec: np.ndarray


type TokenEmbeddings = list[TokenEntry]

type Embeddings = dict[str, TokenEmbeddings]
