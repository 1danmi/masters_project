import os

import copy
from typing import Any

import torch
import numpy as np
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

import shelve
from bert2vec.model.Row import Row
from bert2vec.data_process.tf_idf import get_bow_idf_dict

RADIUS = 0.62
ACCEPT_AT = 0.69
PATH = "D:/Studies/Phd/bert/Bert2Vec/Vocab/New"
TABLE_DIR = "shelve"
TABLE = "table.slv"


stop_words = set(stopwords.words("english"))


# get bag of words of the ith word in the list
def getBagOfWords(text: list, i):
    return text[i - 5 if i > 5 else 0 : i] + text[i + 1 : i + 6]


class Model:
    def __init__(self, path=f"{PATH}\\{TABLE_DIR}\\{TABLE}"):
        self.path = path
        self.dirPath = f"{PATH}\\{TABLE_DIR}"
        self.rows = {}
        self.last_tokenID = 0

    @staticmethod
    def isTableExists(path):
        return os.path.isfile(f"{path}.dat")

    # return vec by the most count
    def get_vec(self, word: str):
        rows = self.rows.get(word)
        if len(rows) == 0:
            return np.zeros(768)
        elif len(rows) == 1:
            return rows[0].vec
        else:
            return max(rows, key=lambda row: row.count).vec

    # def getRowsByBert(self, token: str, sentence: str):
    #     # additions for new retrieval function bert based.
    #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #     model = BertModel.from_pretrained('bert-base-uncased',
    #                                            output_hidden_states=True)
    #     """Retrieve rows for a token based on cosine similarity to BERT embeddings."""
    #     inputs = tokenizer(sentence, return_tensors='pt')
    #
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #
    #     hidden_states = outputs.last_hidden_state
    #
    #     token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    #
    #     try:
    #         token_index = token_ids.index(tokenizer.convert_tokens_to_ids(token))
    #     except ValueError:
    #         raise ValueError(f"Token '{token}' not found in the sentence '{sentence}'")
    #
    #     bert_vector = hidden_states[0, token_index, :].numpy()
    #
    #     results = []
    #
    #     if token in self.rows:
    #         for row in self.rows[token]:
    #             similarity = cosine_similarity([bert_vector], [row.vec])[0][0]
    #             results.append((row, similarity))
    #
    #     sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    #
    #     return sorted_results

    def getRowsByBOW_TFIDF(self, word, BOW: set):

        rows = self.rows.get(word)
        BOW_IDF_dict = get_bow_idf_dict(BOW, rows)
        # create a tuple of (Row, BM25_rank)
        lst = []
        for r in rows:
            lst += [
                (
                    r,
                    sum(
                        BOW_IDF_dict.get(w) *  # IDF
                        # (
                        r.BOW.get(w, 0)
                        # / r.count )
                        / sum(r.BOW.values())  # TF
                        for w in BOW
                    ),
                )
            ]
        return sorted(lst, key=lambda t: t[1], reverse=True)

        # recevies a word, returns a list of most appropriate rows according to BM25 formula

    # def getRowsByBOW_BM25(self, word, BOW: set, k1=1.5):
    #
    #     rows = self.rows.get(word)
    #     BOW_IDF_dict = get_bow_idf_dict(BOW, rows)
    #     # create a tuple of (Row, BM25_rank)
    #     lst = []
    #     for r in rows:
    #         lst += [(r, sum(
    #             BOW_IDF_dict.get(w) *  # IDF
    #             ((r.BOW.get(w, 0) * (k1 + 1))
    #              /
    #              (r.BOW.get(w, 0) + k1))  # * (1 - b + b * (sum(r.BOW.values()) / avgdl))))
    #             for w in BOW
    #         ))]
    #     return sorted(lst, key=lambda t: t[1], reverse=True)

    def getVecByBow(self, word: str, BOW: set):
        row = self.getRowByBow(word, BOW)
        return row.vec

    def getRowByBow(self, word: str, BOW: set):
        lst = self.getRowsByBOW_BM25(word, BOW)
        return lst[0][0]

    # receives a word and n, returns the n entries (Rows) of word, sorted descending by count
    # Bag of Words is also sorted descending
    # the returned result is list of (origEntry, Row)
    def getRowsByCount(self, word, n=5):
        if not word in self.rows:
            return []
        rows = copy.deepcopy(self.rows[word])
        ret = []
        for i in range(len(rows)):
            ret += [(i, rows[i])]
        # sort by count
        ret = sorted(ret, key=lambda t: t[1].count, reverse=True)
        # sort each BOW by count
        for r in ret:
            r[1].BOW = dict(sorted(r[1].BOW.items(), key=lambda x: x[1], reverse=True))

        return ret[:n]

    # # s - word, vec - the Bert vector, Bow - the environment of the word
    # # adds this entry to our vocab
    # def add_word(self, s: str, vec, BOW):
    #     rows_list, row, dist = self.if_word_exists(s, vec)
    #     if rows_list is not None:  # if word exists in our vocab
    #         if dist > ACCEPT_AT:  # if the distance is above acceptance
    #             row.update_vec_count(vec)
    #             row.update_BOW(BOW)
    #         elif dist < RADIUS:  # if distance is below rejection - new entry
    #             rows_list.append(Row({x: 1 for x in BOW}, s, vec))
    #     else:
    #         self.rows[s] = [Row({x: 1 for x in BOW}, s, vec)]
    #
    # # receives a word, returns a tuple [isExists, row, dist]
    # # isExists - whether the word exists in the dict
    # # row - the row with maximum ???
    # # max_cos - the maximum cosine similarity
    # def if_word_exists(self, s: str, vec) -> tuple[Any, Any, Any]:
    #     rows_list = self.rows.get(s)
    #     if rows_list is None: return None, None, None
    #     max_row = None
    #     max_cos = -1
    #     for row in rows_list:
    #         # cos = row.cosine_vectorized_v3(vec)
    #         cos = cosine_similarity(np.reshape(row.vec, (1, -1)), np.reshape(vec, (1, -1)))
    #         if cos > max_cos:
    #             max_cos = cos
    #             max_row = row
    #     return rows_list, max_row, max_cos

    # def save_table(self):
    #     if not os.path.exists(f"{PATH}\\{TABLE_DIR}"):
    #         os.makedirs(f"{PATH}\\{TABLE_DIR}")
    #     with shelve.open(self.path, writeback=True) as s:
    #         s.update(self.rows)
    #         s.sync()

    # @staticmethod
    # def read_data(path=f"{PATH}\\{TABLE_DIR}\\{TABLE}", inMem=False):
    #     table = Model()
    #     table.inMem = inMem
    #     if not Model.isTableExists(path): return table
    #     if not inMem:  # no need to close shelve, just open from HD as dict
    #         table.rows = shelve.open(path)
    #     else:
    #         with shelve.open(path) as s:
    #             table.rows = dict(s)  # read whole
    #     return table

    def search_str(self, s: str):
        return self.rows.get(s)

    def find_close_n_similar_count(
        self,
        s,
        n: int,
        include_partial_words=True,
        include_self=False,
        include_add_sub=False,
        include_duplicates=True,
        add_words=[],
        sub_words=[],
    ):
        maxes = [(None, float("-inf"))] * n
        row = max(self.rows.get(s), key=lambda row: row.count)
        for w in add_words:
            row.vec = np.add(row.vec, max(self.rows.get(w), key=lambda row: row.count).vec)
        for w in sub_words:
            row.vec = np.subtract(row.vec, max(self.rows.get(w), key=lambda row: row.count).vec)
        for word in self.rows.values():
            for r in word:
                if include_partial_words is False and r.s.startswith("##"):
                    continue
                if r.s == row.s:
                    if not include_self:
                        continue
                if r.s in add_words or r.s in sub_words:
                    if not include_add_sub:
                        continue
                if include_duplicates is False and r.s in [max[0].s for max in maxes if max[0] is not None]:
                    continue
                i, (_, m) = min(enumerate(maxes), key=lambda q: q[1][1])
                cos = cosine_similarity(np.reshape(row.vec, (1, -1)), np.reshape(r.vec, (1, -1)))
                if m < cos:
                    maxes[i] = (r, cos)
        return sorted(maxes, key=lambda k: k[1], reverse=True)

    def find_close_n_similar(
        self,
        row: Row,
        n: int,
        include_partial_words=True,
        include_self=False,
        include_add_sub=False,
        include_duplicates=True,
        add_words=[],
        sub_words=[],
    ):
        maxes = [(None, float("-inf"))] * n
        # from copy import deepcopy
        # row = deepcopy(row)
        for w in add_words:
            row.vec = np.add(row.vec, max(self.rows.get(w), key=lambda row: row.count).vec)
        for w in sub_words:
            row.vec = np.subtract(row.vec, max(self.rows.get(w), key=lambda row: row.count).vec)

        for word in self.rows.values():
            for r in word:
                if include_partial_words is False and r.s.startswith("##"):
                    continue
                if r.s == row.s:
                    if not include_self:
                        continue
                if r.s in add_words or r.s in sub_words:
                    if not include_add_sub:
                        continue
                if include_duplicates is False and r.s in [max[0].s for max in maxes if max[0] is not None]:
                    continue
                i, (_, m) = min(enumerate(maxes), key=lambda q: q[1][1])
                cos = cosine_similarity(np.reshape(row.vec, (1, -1)), np.reshape(r.vec, (1, -1)))
                if m < cos:
                    maxes[i] = (r, cos)
        return sorted(maxes, key=lambda k: k[1], reverse=True)
