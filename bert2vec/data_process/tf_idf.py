from math import log2

#returns the BOW of entire row (word entry)
def getBOWSet(row: list):
    bow = set()
    for r in row:
        bow = bow.union(r.BOW)
    return bow

#returns the number of rows containing word in BOW
#rows = count of the BOW appeard for each row
def getDocumentCount(word : str, rows : list):
    return len([r.BOW.get(word,None) for r in rows])

#returns a dictionary of word : IDF(word,rows)
def get_bow_idf_dict(BOW : set, rows : list):
    # return {i : getWordIDF(i,rows) for i in BOW}
    N = sum([r.count for r in rows])
    d = dict()
    for i in BOW:
        wc = getDocumentCount(i, rows)
        d[i] = 0 if wc == 0 else log2(N/wc)

    return d