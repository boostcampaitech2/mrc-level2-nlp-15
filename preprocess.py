import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset

# reference: https://github.com/snoop2head/Mathpresso_Classification/blob/main/modules/preprocess_for_kobert.py
def preprocess(text):
    text = re.sub(r"\\r|\\n|\n|\\t", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_json(json_data: dict) -> dict:
    json_data = json_data["text"].apply(lambda x: preprocess(x))
    return json_data


def preprocess_df(df: pd.DataFrame):
    df["text"] = df["text"].apply(lambda x: preprocess(x))
    return df


def preprocess_individual_dict(single_item: dict) -> dict:
    """preprocess individual dictionary from huggingface dataset"""

    # fetch from single_item datasets
    answer_start = single_item["answers"]["answer_start"][0]
    answer_text = single_item["answers"]["text"][0]
    context = single_item["context"]

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
    single_item["context"] = new_context
    single_item["answers"]["answer_start"][0] = new_answer_start

    return single_item


def preprocess_train_val(train_val: dict) -> list:
    """make preprocessed new train_val from datasets['train'] or datasets['validation'] from huggingface dataset"""
    new_train_val = list()

    for idx, single_item in tqdm(enumerate(train_val)):
        new_train_val.append(preprocess_individual_dict(single_item))

    new_train_val_dataset = Dataset.from_pandas(pd.DataFrame(new_train_val))
    return new_train_val_dataset
