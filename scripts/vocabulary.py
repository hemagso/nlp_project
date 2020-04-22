import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import csv
from collections import Counter
from tqdm import tqdm


def list_words(review):
    def token_generator(review):
        for sentence in sent_tokenize(review):
            for word in word_tokenize(sentence):
                yield word.lower()
    return Counter(token_generator(review))


nltk.download('punkt')

df_reviews = pd.read_csv("../data/reviews/reviews_process.csv")
df_en = df_reviews.loc[df_reviews["language"] == "en"].copy()

print(df_reviews.shape, df_en.shape)

del df_reviews

vocabulary = Counter()

for review in tqdm(df_en["review"]):
    vocabulary += list_words(review)

with open("../data/reviews/vocabulary.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(("word", "count"))
    for word, count in tqdm(vocabulary.items()):
        writer.writerow((word, count))

