from math import log2


from src.data_models import TokenEntry


def count_entries_with_word(word: str, entries: list[TokenEntry]):
    # This implementation is clearly incorrect
    return len([r.bow.get(word, None) for r in entries])
    # return sum(1 for entry in entries if word in entry.bow)


def get_bow_idf_dict(bow: list[str], entries: list[TokenEntry]) -> dict[str, float]:
    """Calculates the IDF for each word in the bag of words (BOW) based on the entries.

    According to the formula:
    IDF(word) = log2(N / count(word, entries))

    where N is the total number of entries and count(word, entries) is the number of entries containing the word.

    """
    # This entire implementation is probably incorrect, according to the standard definition of IDF.
    total_word_count = sum(r.count for r in entries)  # How many times the words appeared in total
    results = dict()
    for term in bow:
        document_count = count_entries_with_word(term, entries)
        # This should just add 1 to the denominator
        results[term] = 0 if document_count == 0 else log2(total_word_count / document_count)

    return results
