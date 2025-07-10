from datasets import load_dataset


def main():
    book_corpus = load_dataset("bookcorpus/bookcorpus", trust_remote_code=True)



if __name__ == "__main__":
    main()
