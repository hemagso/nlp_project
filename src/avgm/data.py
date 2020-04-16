import csv


class Vocabulary(object):
    SPECIAL_TOKENS = ["<PAD>", "<UNK>"]
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, vocab_path):
        self._load_csv_file(vocab_path)

    def _load_csv_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skipping header
            self.id2vocab = {idx + len(Vocabulary.SPECIAL_TOKENS): word for idx, (word, _) in enumerate(reader)}
            self.vocab2id = {word: idx for idx, word in self.id2vocab.items()}
        self._add_special_tokens()

    def _add_special_tokens(self):
        for idx, token in enumerate(Vocabulary.SPECIAL_TOKENS):
            self.id2vocab[idx] = token
            self.vocab2id[token] = idx

    def to_vocab(self, id_list):
        return [self.id2vocab[idx] for idx in id_list]

    def to_idx(self, token_list):
        return [self.vocab2id.get(token.lower(), Vocabulary.UNK_IDX) for token in token_list]


if __name__ == "__main__":
    vocab = Vocabulary("../../data/reviews/vocabulary_test.csv")
    print(vocab.to_idx(["I", "am", "crazy", "about", "this", "game", "!"]))


