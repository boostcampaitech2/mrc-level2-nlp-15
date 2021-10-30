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


def preprocess_train_val(train_val: dict) -> dict:
    """train_val such as datasets['train'] or datasets['validation'] from huggingface dataset"""
    # fetch from train_val datasets
    answer_start = train_val["answers"]["answer_start"][0]
    answer_text = train_val["answers"]["text"][0]
    context = train_val["context"]

    # slice the context
    preceding_text = context[:answer_start]
    following_text = context[answer_start + len(answer_text) :]

    # apply preprocessing for each sliced context
    preprocessed_preceding = preprocess(preceding_text)
    preprocessed_following = preprocess(following_text)

    # make preprocessed dataset dictionary
    new_context = preprocessed_preceding + answer_text + preprocessed_following
    new_answer_start = answer_start - abs(len(preceding_text) - len(preprocessed_preceding))

    # assign preprocessed values
    train_val["context"] = new_context
    train_val["answers"]["answer_start"][0] = new_answer_start

    return train_val
