import numpy as np
from collections import Counter


class Row:
    def __init__(self, BOW: dict, s, vec, count=1):
        self.BOW = BOW  # Bag of words, environment of the word, each entry is a {word: count}
        self.s = s  # the string
        self.vec = vec  # vector
        self.count = count  # counter
        self.tokenID = 0

    # recievs a list of tuples (vec, count) and summerizes all vectors, averaging the count
    # returns tuple of vec, count
    @staticmethod
    def sumVecs(vec_count: list):
        s = sum([t[1] for t in vec_count])
        arr = np.array([np.multiply(t[0], t[1] / s) for t in vec_count])

        return arr.sum(axis=0)

    ###vec - vector from Bert, count - count of vector
    # computes the average of the row's current vector and another Bert vector
    def update_vec_count(self, vec, count=1):
        self.vec = Row.sumVecs([(self.vec, self.count), (vec, count)])
        self.count = self.count + count

    def update_BOW(self, bow):
        for w in bow:
            if w in self.BOW.keys():
                self.BOW[w] += 1
            else:
                self.BOW[w] = 1

    def __repr__(self):
        return f"{self.s!r},{self.vec!r},{self.BOW!r},{self.count!r}"

    def uniteRows(self, row):
        self.update_vec_count(row.vec, row.count)
        self.BOW = dict(Counter(self.BOW) + Counter(row.BOW))
