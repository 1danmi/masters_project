import pandas as pd
import numpy as np
from bert2vec.data_process.tf_idf import get_bow_idf_dict


# returns the average length of row in terms of BOW
def getBOW_Average(rows: list):
    s = 0
    for r in rows:
        s += sum(r.BOW.values())
    return s / len(rows)


def get_BM25_Mat(BOW: set, rows: list, k1):
    BOW_IDF_dict = get_bow_idf_dict(BOW, rows)
    df = pd.DataFrame(columns=list(BOW))
    # avgdl = getBOW_Average(rows)
    for r in rows:
        df = pd.concat(
            [
                df,
                pd.DataFrame.from_records(
                    [
                        {
                            w: BOW_IDF_dict.get(w)
                            * (
                                (r.BOW.get(w, 0) * (k1 + 1) / r.count) / (r.BOW.get(w, 0) / r.count + k1)
                            )  # * (1 - b + b * (sum(r.BOW.values()) / avgdl))))
                            for w in BOW
                        }
                    ]
                ),
            ]
        )
        # sum(r.BOW.values()) - length of row in BOW
    return df


def getMaxVectorBM25(word: str, BOW: set, table, k1, b):
    rows = table.rows.get(word)
    if rows is None:
        return np.zeros(768)  # TODO : change this to a variable
    BOW_IDF_dict = get_bow_idf_dict(BOW, rows)
    avgdl = getBOW_Average(rows)
    maxBM25 = 0
    maxRow = None
    for row in rows:
        s = sum(
            [
                BOW_IDF_dict.get(w)
                * (
                    (row.BOW.get(w, 0) * (k1 + 1) / row.count)
                    / (row.BOW.get(w, 0) / row.count + k1 * (1 - b + b * (sum(row.BOW.values()) / avgdl)))
                )
                for w in BOW
            ]
        )
        if s > maxBM25:
            maxBM25, maxRow = s, row

    return maxRow
