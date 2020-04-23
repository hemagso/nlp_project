import h5py
import json
import bpe
from torch.utils import data
import torch.nn as nn
import torch

class Dataset(data.Dataset):
    def __init__(self, hdf5_dataset, dstype="train"):
        self.dstype = dstype
        self._load_data(hdf5_dataset)
        self._load_metadata(hdf5_dataset)
        assert len(self.tokens) == len(self.scores)

    def _load_data(self, hdf5_dataset):
        self.tokens = [
            torch.LongTensor(item) for item in hdf5_dataset["data/{dstype}/tokens".format(dstype=self.dstype)]
        ]
        self.scores = [
            torch.LongTensor([item]) for item in hdf5_dataset["data/{dstype}/scores".format(dstype=self.dstype)]
        ]

    def _load_metadata(self, hdf5_dataset):
        vocab_dict = json.loads(hdf5_dataset["metadata/encoder"][()])
        self.encoder = bpe.Encoder.from_dict(vocab_dict)
        self.PAD_INDEX = self.encoder.word_vocab[self.encoder.PAD]
        self.UNK_INDEX = self.encoder.word_vocab[self.encoder.UNK]

    def __getitem__(self, item):
        return self.tokens[item], self.scores[item]

    def __iter__(self):
        return iter(zip(self.tokens, self.scores))

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def read_hdf5(path, datasets=("train", "valid", "test")):
        ret = ()
        with h5py.File(path, "r") as f:
            for dataset in datasets:
                ret += (Dataset(f, dstype=dataset),)
        return ret


class PadSequence(object):
    def __init__(self, padding_value):
        self.padding_value = padding_value

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sequences = [tokens for tokens, score in sorted_batch]
        lengths = torch.LongTensor([len(tokens) for tokens in sequences])
        scores = torch.LongTensor([scores for tokens, scores in sorted_batch])
        sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.padding_value)
        return sequences_padded, lengths, scores


