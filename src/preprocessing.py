from src.miscellaneous import *

from nltk.corpus import stopwords
from pandas import DataFrame
from spacy.lang.es.stop_words import STOP_WORDS as spacy_stop_words

import nltk
import os
import pandas as pd
import pickle
import spacy

nltk.download('stopwords')
try:
    spacy.load('es_core_news_md')
except OSError:
    print("Spacy module will be downloaded.")
    spacy.cli.download('es_core_news_md')


def nltk_exclusive_stopwords():
    """
    nltk_exclusive_stopwords finds stop words that are in the nltk library but not in spacy.

    :return: a sorted list of nltk exclusive stop words
    """

    nltk_stop_words = set(stopwords.words('spanish'))
    spacy_stop_words = set(spacy.lang.es.stop_words.STOP_WORDS)

    return sorted(list(nltk_stop_words - spacy_stop_words))

def spacy_exclusive_stopwords():
    """
    spacy_exclusive_stopwords finds stop words that are in the spacy library but not in nltk.

    :return: a sorted list of spacy exclusive stop words
    """

    nltk_stop_words = set(stopwords.words('spanish'))
    spacy_stop_words = set(spacy.lang.es.stop_words.STOP_WORDS)

    return sorted(list(spacy_stop_words - nltk_stop_words))

def union_exclusive_stopwords():
    """
    union_exclusive_stopwords finds the union of stop words that are exclusive to either the nltk or spacy libraries.

    :return: a sorted list of the union of nltk and spacy exclusive stop words
    """

    nltk_exclusive = set(nltk_exclusive_stopwords())
    spacy_exclusive = set(spacy_exclusive_stopwords())

    return sorted(list(nltk_exclusive.union(spacy_exclusive)))

def preprocess_data(saved_data, dataset, df) -> DataFrame:
    """
    preprocess_data is used to perform the preprocess step according to the dataset used

    :param saved_data: indicates in case there exists a saved intermediate state to use it or not
    :param dataset: the name of the dataset in use
    :return: 
    """

    if saved_data == False:
        if dataset == "UPV":
            preprocess_UPV()
        elif dataset == "Wikipedia":
            return preprocess_Wikipedia()
    else:
        if dataset == "UPV":
            if os.path.exists('mediaUPV-preprocessed.pickle'):
                return load_pickle()
            else:
                return preprocess_UPV()
        elif dataset == "Wikipedia":
            if os.path.exists('wikipedia-preprocessed.pickle'):
                return load_pickle()
            else:
                return preprocess_Wikipedia()

def preprocess_Wikipedia():
    """
    preprocess_Wikipedia is used to preprocess Wikipedia dataset

    :return: preprocessed DataFrame
    """

    # Load the Wikipedia dataset
    df = load_wikipedia_dataset()
    
    # Clean the dataset
    clean_dataset(df)
    
    # Exclude entries with empty categories
    df = exclude_empty_categories('Wikipedia', df)
    
    # Preprocess text for each entry in the dataset
    df['text'] = df['text'].apply(preprocess_text)
    
    # Save preprocessed DataFrame for future use
    save_pickle(df, 'wikipedia-preprocessed', f'{get_root_projet()}{wikipedia_path}')
    
    return df

def preprocess_UPV():
    """
    preprocess_UPV is used to preprocess UPV dataset

    :return: preprocessed DataFrame
    """

    # Load the UPV dataset
    df = load_upv_dataset()
    
    # Clean the dataset
    clean_dataset(df)
    
    # Preprocess text for each entry in the dataset
    df['text'] = df['text'].apply(preprocess_text)

    # Convert 'Keywords' column from string to list
    df = convert_string_to_list(df, 'Keywords')
    
    # Save preprocessed DataFrame for future use
    save_pickle(df, 'mediaUPV-preprocessed', f'{get_root_projet()}{upv_path}')
    
    return df

def clean_dataset(df):
    """
    clean_dataset takes in a pandas DataFrame and removes any rows containing null values (NaNs),
    duplicate rows, and resets the index.

    :param df: A pandas DataFrame to be cleaned.
    :return: The function does not return anything. It modifies the DataFrame in-place.
    """

    df.dropna(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

def get_dataset_info(df):
    """
    get_dataset_info returns information about a DataFrame including the index dtype and column dtypes, 
    non-null values and memory usage.

    :param df: A pandas DataFrame.
    :return: Information about DataFrame including the index dtype and column dtypes, 
             non-null values and memory usage.
    """

    return df.info()

def get_dataset_describtion(df):
    """
    get_dataset_describtion generates descriptive statistics of a DataFrame. 
    This includes central tendency, dispersion and shape of the dataset’s distribution, excluding NaN values.

    :param df: A pandas DataFrame.
    :return: Descriptive statistics of the DataFrame.
    """

    return df.describe()

def preprocess_text(text):
    """
    preprocess_text preprocesses the input text by performing tokenization, lemmatization,
    and filtering out stop words, non-alphabetical characters, and words with length less or equal to 3.

    :param text: A string of text to be preprocessed.
    :return: A list of processed tokens.
    """

    # Tokenize and lemmatize
    doc = nlp(text)
    processed_tokens = [token.lemma_ for token in doc if not token.is_stop]
    # Filter out non-alphabetical characters and words with length <= 3
    processed_tokens = [token for token in processed_tokens if re.match(r'^[a-zA-Z]+$', token) and len(token) > 3]
    return processed_tokens

def exclude_empty_categories(dataset, df):
    """
    exclude_empty_categories excludes rows with empty categories from the dataset. 
    It currently only supports the "Wikipedia" dataset.

    :param dataset: The name of the dataset. Currently, only "Wikipedia" is supported.
    :param df: A pandas DataFrame.
    :return: A DataFrame with rows having empty categories excluded.

    The function raises a ValueError if a dataset other than "Wikipedia" is provided.
    """

    if dataset == "Wikipedia":
        return train_df[train_df['categories'] != "[]"]
    else:
        raise ValueError(f"Dataset should be Wikipedia, not {dataset}")

def convert_string_to_list(df, column):
    """
    convert_string_to_list converts a DataFrame column of strings into a list of strings, splitting each string by spaces.

    :param df: Input DataFrame
    :param column: Column name to process
    :return: DataFrame with the processed column
    """

    df[column] = df[column].str.split()
    return df