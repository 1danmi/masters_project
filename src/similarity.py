import json

from src import Bert2VecModel
from src.config import config


def f(word: str):
    with Bert2VecModel(source_path=config().dest_path, in_mem=False) as m:
        first = sorted(m[word][0].bow.items(), key=lambda item: -item[1])
        second = dict(sorted(m[word][1].bow.items(), key=lambda item: -item[1]))
        count = 0
        for word, freq in first:
            if second.get(word, 0) > freq:
                count += 1
                print(f"Word: {word}:\tfirst: {freq}\tsecond: {second.get(word)}")
                if count == 1000:
                    break


def check_word_pairs():
    with Bert2VecModel(source_path=config().dest_path, in_mem=False) as model:
        print(f"Finished loading model")
        with open("src/pairs2.json", "r") as f:
            pairs = json.load(f)
        for word, bags in pairs.items():
            idx1 = model.get_entry_idx_by_bow(token=word, bow=bags[0])
            idx2 = model.get_entry_idx_by_bow(token=word, bow=bags[1])
            print(f"word: {word}\t({', '.join(bags[0])}): {idx1}\t({', '.join(bags[1])}):\t{idx2}")


if __name__ == "__main__":
    check_word_pairs()
