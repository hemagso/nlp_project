import langdetect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import csv


def classify_language(text):
    try:
        return langdetect.detect(text)
    except LangDetectException as e:
        return "??"


df_reviews = pd.read_csv("../data/reviews/reviews.csv", encoding="utf-8")
df_reviews["language"] = df_reviews["review"].apply(classify_language)
df_reviews["length"] = df_reviews["review"].apply(len)

df_reviews.to_csv("../data/reviews/reviews_process.csv", quoting=csv.QUOTE_ALL, index=False, encoding="utf-8")
