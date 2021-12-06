"""
Loading dataset
:author: Qizhi Li
"""
import os
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer

from static_data import file_path as fp


def load_data_frame(file_path):
    """
    Loading DataFrame
    :return: DataFrame
    """
    return pd.read_csv(file_path, delimiter='\t', header=None)


def get_binary_data(df):
    """
    Obtaining binary sentiment classification data
    :param df: DataFrame
    :return df: DataFrame
            cleaned data
    """
    df = df.drop(df[df[2] == 'neutral'].index, axis=0)
    df = df.drop(df[df[2] == 'objective-OR-neutral'].index, axis=0)
    df = df.drop(df[df[2] == 'objective'].index, axis=0)
    return df


def clean_semeval_data(df):
    """
    Clean up SemEval data
    :param df: DataFrame
    :return cleaned_df: DataFrame
    """

    df = get_binary_data(df)
    texts = df[3].values
    orientation = df[2].values
    orientation_int = np.array([], dtype=int)
    for o in orientation:
        if o == 'positive':
            orientation_int = np.append(orientation_int, 1)
        else:
            orientation_int = np.append(orientation_int, 0)
    cleaned_df = pd.DataFrame({'texts': texts, 'orientation': orientation_int})

    return cleaned_df


def change_semeval_to_binary():
    """
    Changing SemEval dataset into binary sentiment classification task, include:
        1. Only extracting texts and their sentiment orientation,
           and separated them by \t;
        2. delete objective, neutral, and objective-OR-neutral;
        3. change positive into 1, and negative into 0.
    :return:
    """
    train_data = load_data_frame(fp.semeval_train)
    dev_data = load_data_frame(fp.semeval_dev)
    test_data = load_data_frame(fp.semeval_test)

    cleaned_train = clean_semeval_data(train_data)
    cleaned_dev = clean_semeval_data(dev_data)
    cleaned_test = clean_semeval_data(test_data)

    cleaned_train.to_csv(fp.cleaned_semeval_train, sep='\t', header=False, index=False)
    cleaned_dev.to_csv(fp.cleaned_semeval_dev, sep='\t', header=False, index=False)
    cleaned_test.to_csv(fp.cleaned_semeval_test, sep='\t', header=False, index=False)


def get_Xy(df):
    """
    Obtaining the X and y from tsv file
    :param df: DataFrame
    :return X: ndarray
            texts
    :return y: ndarray
            labels
    """
    X = df[0].values
    y = df[1].values

    return X, y


def rem_urls(tokens):
    final = []
    for t in tokens:
        if t.startswith('@') or t.startswith('http') or t.find('www.') > -1 or t.find('.com') > -1:
            pass
        elif t[0].isdigit():
            final.append('NUMBER')
        else:
            final.append(t)
    return final


def load_data(dataset, tokenize=True):
    """
    Loading experiment dataset
    :param dataset: str
            'SST'
            'semeval'
    :param tokenize: bool
            True: tokenize sentences
            False: keep the original sentence
    :return X_train: ndarray (tokenize=False) or list (tokenize=True)
    :return y_train: ndarray (tokenize=False) or list (tokenize=True)
    :return X_dev: ndarray (tokenize=False) or list (tokenize=True)
    :return y_dev: ndarray (tokenize=False) or list (tokenize=True)
    :return X_test: ndarray (tokenize=False) or list (tokenize=True)
    :return y_test: ndarray (tokenize=False) or list (tokenize=True)
    """
    if dataset == 'semeval':
        if not os.path.exists(fp.cleaned_semeval_train):
            change_semeval_to_binary()

        train_data = load_data_frame(fp.cleaned_semeval_train)
        dev_data = load_data_frame(fp.cleaned_semeval_dev)
        test_data = load_data_frame(fp.cleaned_semeval_test)

    else:
        train_data = load_data_frame(fp.sst_train)
        dev_data = load_data_frame(fp.sst_dev)
        test_data = load_data_frame(fp.sst_test)

    X_train, y_train = get_Xy(train_data)
    X_dev, y_dev = get_Xy(dev_data)
    X_test, y_test = get_Xy(test_data)

    if tokenize:
        tknzr = TweetTokenizer(preserve_case=True, reduce_len=True, strip_handles=True)
        X_train = [rem_urls(tknzr.tokenize(sent.lower())) for sent in X_train]
        X_dev = [rem_urls(tknzr.tokenize(sent.lower())) for sent in X_dev]
        X_test = [rem_urls(tknzr.tokenize(sent.lower())) for sent in X_test]

    return X_train, y_train, X_dev, y_dev, X_test, y_test
