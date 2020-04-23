import pandas as pd
import bpe
import numpy as np
import h5py
import json

df_train = pd.read_csv("../data/reviews/train.csv", encoding="utf-8")
df_test = pd.read_csv("../data/reviews/test.csv", encoding="utf-8")
df_valid = pd.read_csv("../data/reviews/valid.csv", encoding="utf-8")

encoder = bpe.Encoder(10000, pct_bpe=0.7, EOW="<eow>", SOW="<sow>", UNK="<unk>", PAD="<pad>")
encoder.fit(df_train["review"])
encoder.save("../data/vocabulary/encoder.txt")

train_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_train["review"])))
train_scores = df_train["userscore"].to_numpy(np.dtype("int32"))

valid_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_valid["review"])))
valid_scores = df_valid["userscore"].to_numpy(np.dtype("int32"))

test_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_test["review"])))
test_scores = df_test["userscore"].to_numpy(np.dtype("int32"))

with h5py.File("../data/reviews/tokenized.h5", "w") as f:
    dt = h5py.vlen_dtype(np.dtype("int32"))
    f.create_group("data")
    f.create_group("data/train")
    f.create_dataset("data/train/tokens", data=train_tokens, dtype=dt)
    f.create_dataset("data/train/scores", data=train_scores)
    f.create_group("data/valid")
    f.create_dataset("data/valid/tokens", data=valid_tokens, dtype=dt)
    f.create_dataset("data/valid/scores", data=valid_scores)
    f.create_group("data/test")
    f.create_dataset("data/test/tokens", data=test_tokens, dtype=dt)
    f.create_dataset("data/test/scores", data=test_scores)

    dt = h5py.string_dtype(encoding='utf-8')
    f.create_group("metadata")
    f.create_dataset("metadata/encoder", data=json.dumps(encoder.vocabs_to_dict()), dtype=dt)
