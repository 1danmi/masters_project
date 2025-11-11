import json
from src.config import config
from src import CompactBert2VecModel, Bert2VecModel

if __name__ == "__main__":
    print(f"Loading model...")
    model = CompactBert2VecModel.load(path=config().compact_dest_path)
    print(f"Finished loading model")
    with open("pairs2.json", "r") as f:
        pairs = json.load(f)
    for word, bags in pairs.items():
        token1= model.get_entry_by_bow(word, bags[0])
        token2= model.get_entry_by_bow(word, bags[1])
        print(f"word: {word}\t1: {token1.count}\t2: {token2.count}")
