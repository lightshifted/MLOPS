# tagifai/data.py
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

from config import config


def clean_text(
    text: str, lower: bool = True, stem: bool = False, stopwords=config.STOPWORDS
) -> str:
    """Clean raw text

    Args:
        text (str): _description_
        lower (bool, optional): _description_. Defaults to True.
        stem (bool, optional): _description_. Defaults to False.
        stopwords (_type_, optional): _description_. Defaults to config.STOPWORDS.

    Returns:
        str: _description_
    """
    stemmer = PorterStemmer()
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def replace_oos_labels(
    df: pd.DataFrame, labels: List, label_col: str, oos_label: str = "other"
) -> pd.DataFrame:
    """Replace out-of-scope oos labels

    Args:
        df (pd.DataFrame): _description_
        labels (List): _description_
        label_col (str): _description_
        oos_label (str, optional): _description_. Defaults to "other".

    Returns:
        pd.DataFrame: _description_
    """
    oos_tags = [item for item in df[label_col].unique() if item not in labels]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)
    return df


def replace_minority_labels(
    df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other"
) -> pd.DataFrame:
    """Replace minority labels with another label.

    Args:
        df (pd.DataFrame): _description_
        label_col (str): _description_
        min_freq (int): _description_
        new_label (str, optional): _description_. Defaults to "other".

    Returns:
        pd.DataFrame: _description_
    """

    labels = Counter(df[label_col].values)
    labels_above_freq = Counter(label for label in labels.elements() if (labels[label] >= min_freq))
    df[label_col] = df[label_col].apply(lambda label: label if label in labels_above_freq else None)
    df[label_col] = df[label_col].fillna(new_label)
    return df


class LabelEncoder(object):
    """Encode labels into unique indices.

    Args:
        object (_type_): _description_
    """

    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List):
        """_summary_

        Args:
            y (List): _description_

        Returns:
            _type_: _description_
        """
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y: List) -> np.ndarray:
        """_summary_

        Args:
            y (List): _description_

        Returns:
            np.ndarray: _description_
        """
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y: List) -> List:
        """_summary_

        Args:
            y (List): _description_

        Returns:
            List: _description_
        """
        classes = []
        for i, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, fp: str) -> None:
        """_summary_

        Args:
            fp (str): _description_
        """
        with open(fp, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp: str):
        """_summary_

        Args:
            fp (str): _description_

        Returns:
            _type_: _description_
        """
        with open(fp, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple:
    """Generate balanced data splits.

    Args:
        X (pd.Series): _description_
        y (np.ndarray): _description_
        train_size (float, optional): _description_. Defaults to 0.7.

    Returns:
        Tuple: _description_
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5, stratify=y_)
    return X_train, X_val, X_test, y_train, y_val, y_test


# tagifai/data.py
def preprocess(df: pd.DataFrame, lower: bool, stem: bool, min_freq: int) -> pd.DataFrame:
    """Preprocess the data.

    Args:
        df (pd.DataFrame): _description_
        lower (bool): _description_
        stem (bool): _description_
        min_freq (int): _description_

    Returns:
        pd.Dataframe: _description_
    """
    df["text"] = df.title + " " + df.description  # feature engineering
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # clean text
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # replace labels below min freq

    return df
