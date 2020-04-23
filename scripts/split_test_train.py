import pandas as pd
import numpy as np


def split_dataset(df, proportions):
    assert sum(proportions) == 1
    start = 0
    end = 0
    ans = ()
    for p in proportions:
        end += p
        print(start, end)
        mask = np.random.rand(df.shape[0])
        mask = (start <= mask) & (mask < end)
        ans += (df[mask], )
        start += p
    return ans


df_reviews = pd.read_csv("../data/reviews/reviews_process.csv", encoding="utf-8")
df_en = df_reviews[df_reviews["language"] == "en"].drop("language", axis=1)

proportions = (0.7, 0.15, 0.15)
df_train, df_valid, df_test = split_dataset(df_en, proportions)

df_train.to_csv("../data/reviews/train.csv", encoding="utf-8", index=False)
df_valid.to_csv("../data/reviews/valid.csv", encoding="utf-8", index=False)
df_test.to_csv("../data/reviews/test.csv", encoding="utf-8", index=False)