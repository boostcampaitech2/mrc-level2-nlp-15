import re
import numpy as np
import pandas as pd


# reference: https://github.com/snoop2head/Mathpresso_Classification/blob/main/modules/preprocess_for_kobert.py
def preprocess(text):
    text = re.sub(r"\\r|\\n|\n|\\t", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_json(data):
    text_data = data["text"]
    text_data = text_data.apply(lambda x: preprocess(x))
    return text_data


def preprocess_df(df: pd.DataFrame):
    df["text"] = df["text"].apply(lambda x: preprocess(x))
    return df
