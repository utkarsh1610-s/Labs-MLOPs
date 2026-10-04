import gzip
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split

from snorkel.classification.data import DictDataset, DictDataLoader


def load_clickbait_dataset(
    load_train_labels: bool = False,
    split_dev_valid: bool = False,
    data_dir: str = "data/clickbait",
    seed: int = 123,
):
    """Load the Chakraborty et al. (2016) clickbait headline corpus.

    Label 1 = CLICKBAIT (BuzzFeed, Upworthy, ViralNova, ...),
    label 0 = NEWS (NYT, The Guardian, The Hindu, WikiNews).

    Splits (stratified, fixed seed):
      * test  : 2,000 hand-labeled headlines for final evaluation only
      * valid : 1,000 hand-labeled headlines for model selection
      * train : everything else, labels hidden (-1) unless load_train_labels=True
      * dev   : 200 labeled headlines sampled from train for LF development
    """
    dfs = []
    for fname, label in [("clickbait_data.gz", 1), ("non_clickbait_data.gz", 0)]:
        path = os.path.join(data_dir, fname)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [line.strip() for line in f]
        dfs.append(pd.DataFrame({"text": [l for l in lines if l], "label": label}))

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="text")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_rest, df_test = train_test_split(
        df, test_size=2000, random_state=seed, stratify=df.label
    )
    df_train, df_valid = train_test_split(
        df_rest, test_size=1000, random_state=seed, stratify=df_rest.label
    )
    df_dev = df_train.sample(200, random_state=seed)

    df_train = df_train.reset_index(drop=True)
    if not load_train_labels:
        df_train["label"] = -1

    if split_dev_valid:
        return df_train, df_dev, df_valid.reset_index(drop=True), df_test.reset_index(drop=True)
    return df_train, df_test.reset_index(drop=True)


def get_keras_lstm(num_buckets, embed_dim=16, rnn_state_size=64):
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim))
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu))
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"])
    return lstm_model


def get_keras_cnn(num_buckets, embed_dim=32, num_filters=64, kernel_size=3):
    """1D-CNN text classifier: n-gram-like filters + global max pooling.

    Headlines are short, so a convolution over word windows captures
    clickbait phrases ("you won't believe", "17 things") cheaply.
    """
    cnn_model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(num_buckets, embed_dim),
            tf.keras.layers.Conv1D(num_filters, kernel_size, activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    cnn_model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    return cnn_model


def map_pad_or_truncate(string, max_length=30, num_buckets=30000):
    """Tokenize text, pad or truncate to get max_length, and hash tokens."""
    ids = tf.keras.preprocessing.text.hashing_trick(
        string, n=num_buckets, hash_function="md5"
    )
    return ids[:max_length] + [0] * (max_length - len(ids))


def featurize_df_tokens(df):
    return np.array(list(map(map_pad_or_truncate, df.text)))


def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for i, row in df.sample(frac=1, random_state=2).iterrows():
            transformed_or_none = f(row)
            # If TF returned a transformed example, record it in dict and move to next TF.
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            "TF Name": f.name,
                            "Original Text": row.text,
                            "Transformed Text": transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag-of-words features."""
    ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
    return DictDataLoader(ds, **kwargs)
