import h5py
import json
import bpe


class Dataset(object):
    def __init__(self, hdf5_dataset, dstype="train"):
        self.dstype = dstype
        self._load_data(hdf5_dataset)
        self._load_metadata(hdf5_dataset)
        assert len(self.tokens) == len(self.scores)

    def _load_data(self, hdf5_dataset):
        self.tokens = list(hdf5_dataset["data/{dstype}/tokens".format(dstype=self.dstype)])
        self.scores = list(hdf5_dataset["data/{dstype}/scores".format(dstype=self.dstype)])

    def _load_metadata(self, hdf5_dataset):
        vocab_dict = json.loads(hdf5_dataset["metadata/encoder"][()])
        self.encoder = bpe.Encoder.from_dict(vocab_dict)

    def __getitem__(self, item):
        return self.tokens[item], self.scores[item]

    def __iter__(self):
        return iter(zip(self.tokens, self.scores))

    def __len__(self):
        return len(self.tokens)

    @staticmethod
    def read_hdf5(path, datasets=["train", "valid", "test"]):
        ret = ()
        with h5py.File(path, "r") as f:
            for dataset in datasets:
                ret += (Dataset(f, dstype=dataset),)
        return ret

ds_train, ds_valid, ds_test = Dataset.read_hdf5("../../data/reviews/tokenized.h5")
print(len(ds_train), len(ds_valid), len(ds_test))