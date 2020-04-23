import pandas as pd
import bpe
import numpy as np
import h5py

df_train = pd.read_csv("../data/reviews/train.csv", encoding="utf-8")
df_test = pd.read_csv("../data/reviews/test.csv", encoding="utf-8")
df_valid = pd.read_csv("../data/reviews/valid.csv", encoding="utf-8")

encoder = bpe.Encoder(10000, pct_bpe=0.7)
encoder.fit(df_train["review"])
encoder.save("../data/vocabulary/encoder.txt")

train_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_train["review"])))
train_userscores = df_train["userscore"].to_numpy(np.dtype("int32"))

valid_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_valid["review"])))
valid_userscores = df_valid["userscore"].to_numpy(np.dtype("int32"))

test_tokens = list(map(lambda x: np.array(x, dtype=np.dtype("int32")), encoder.transform(df_test["review"])))
test_userscores = df_test["userscore"].to_numpy(np.dtype("int32"))

with h5py.File("../data/reviews/tokenized.h5", "w") as f:
    dt = h5py.vlen_dtype(np.dtype("int32"))
    f.create_group("train")
    f.create_dataset("train/tokens", data=train_tokens, compression="gzip", compression_opts=9, dtype=dt)
    f.create_dataset("train/scores", data=train_userscores, compression="gzip", compression_opts=9)
    f.create_group("valid")
    f.create_dataset("valid/tokens", data=valid_tokens, compression="gzip", compression_opts=9, dtype=dt)
    f.create_dataset("valid/scores", data=valid_userscores, compression="gzip", compression_opts=9)
    f.create_group("test")
    f.create_dataset("test/tokens", data=test_tokens, compression="gzip", compression_opts=9, dtype=dt)
    f.create_dataset("test/scores", data=test_userscores, compression="gzip", compression_opts=9)