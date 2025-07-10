import numpy as np

from bert2vec.model.Row import Row
from bert2vec.model.Model import Model

# bert2vec_path = 'D:/Studies/Phd/bert/Bert2Vec/Vocab/new-stopwords/shelve.slv'
bert2vec_path = "C:/Users/danie/PycharmProjects/Final Project/shared_files/shelve-unite/shelve.slv"
bert2vec_model = Model.read_data(path=bert2vec_path, inMem=False)

bank = bert2vec_model.rows["bank"]

river_bank = bert2vec_model.get_row_by_bow("bank", {"water", "mud", "river"})
river_bank_Bert = bert2vec_model.get_rows_by_bert("bank", "the river bank has water stream")

result = bert2vec_model.get_rows_by_count("bank", 5)


king = bert2vec_model.get_row_by_bow("king", {"palace", "crown", "royal"})
man = bert2vec_model.get_row_by_bow("man", {"person", "son", "child"})
woman = bert2vec_model.get_row_by_bow("woman", {"daughter", "girl", "jewelry"})
tmp = np.subtract(np.add(king.vec, woman.vec), man.vec)
sim = bert2vec_model.find_close_n_similar(Row({}, s=None, vec=tmp), 10, include_partial_words=False)

print(
    f"""
    {king=}
    {man=}
    {woman=}
    {tmp=}
    {sim=}
    """
)
