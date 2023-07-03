from collections import Counter
from langdetect import detect
from pandas import DataFrame
from spellchecker import SpellChecker
from tqdm import tqdm
from translate import Translator

import ast
import os
import pandas as pd
import pickle
import spacy


def get_root_projet() -> str:
    """
    get_root_projet returns the project path

    :return: project path in a string object type
    """

    return os.getcwd()

def translate_text(text, max_length=500):
    """
    translate_text is used to translate the given text from Spanish to English.

    :param text: The text to be translated.
    :param max_length: The maximum length for a chunk of text to be translated at once (default: 500).
    :return: The translated text.

    Note: This function splits the input text into chunks of up to max_length characters, translates each chunk separately,
    and then concatenates the translated chunks.
    """
    
    translator = Translator(from_lang='es', to_lang='en')

    parts = []
    while len(text) > max_length:
        part = text[:max_length]
        last_space = part.rfind(' ')
        if last_space > -1:
            part = part[:last_space]
        parts.append(part)
        text = text[len(part):]
    parts.append(text)

    return ' '.join(translator.translate(p) for p in parts)


def iterative_uniques(df, column):
    """
    iterative_uniques extracts all unique words from a specific column in a pandas DataFrame.

    :param df: The DataFrame to process.
    :param column: The name of the column to extract unique words from.
    :return: A list of unique words.

    Note: The function assumes that each cell in the DataFrame contains a string of space-separated words.
    It splits each string into a set of words, then iteratively merges these sets to find all unique words.
    """

    # Convert the DataFrame column to a list of sets of words
    data = df[column].apply(lambda x: x.split()).apply(set).tolist()

    while len(data) > 1:
        new_sets = []
        for i in range(0, len(data), 2):
            if i < len(data) - 1:
                new_set = data[i].union(data[i+1])
            else:
                new_set = data[i]
            new_sets.append(new_set)
        data = new_sets

    # Convert set to list
    unique_words = list(data[0])
    return unique_words

def detect_non_spanish_words(words):
    """
    detect_non_spanish_words takes a list of words and identifies the ones which are not in Spanish.

    :param words: List of words
    :return: Two lists: non-Spanish words and words for which language detection failed.
    """

    non_spanish_words = []
    exception_non_spanish_words = []

    for word in tqdm(words):
        try:
            if detect(word) != 'es':
                non_spanish_words.append(word)
        except:
            exception_non_spanish_words.append(word)
    
    return non_spanish_words, exception_non_spanish_words

def spellchecker_correct_word(word):
    """
    spellchecker_correct_word takes a word as input and returns the most probable correct spelling
    according to a Spanish spellchecker.

    :param word: Input word as a string
    :return: Most probable correct spelling of the input word
    """

    spell = SpellChecker(language='es')
    return spell.correction(word)

def is_word_in_vocab(word):
    """
    is_word_in_vocab takes a word as input and checks if it's in the SpaCy vocabulary for Spanish.

    :param word: Input word as a string
    :return: True if the word is in the vocabulary, False otherwise
    """

    nlp = spacy.load('es_core_news_lg')
    return word in nlp.vocab.strings

def filter_words(word_list, filter_list):
    """
    filter_words takes a list of words and a filter list. 
    It returns a new list containing only the words from word_list that are not in filter_list.

    :param word_list: List of words to be filtered
    :param filter_list: List of words to be excluded
    :return: Filtered list of words
    """

    return [word for word in word_list if word not in filter_list]

def split_into_words(text):
    """
    split_into_words takes a string of text and splits it into a list of words.

    :param text: Input text as a string
    :return: List of words in the text
    """

    return text.split()


def filter_out_words(word_list, filter_list):
    """
    filter_out_words takes a list of words and a filter list. 
    It returns a new list containing only the words from word_list that are not in filter_list.

    :param word_list: List of words to be filtered
    :param filter_list: List of words to be excluded
    :return: Filtered list of words
    """

    return [word for word in word_list if word not in filter_list]

def join_words(word_list):
    """
    join_words takes a list of words and joins them into a single string with spaces.

    :param word_list: List of words
    :return: Single string with all the words joined by spaces
    """

    return ' '.join(word_list)

def remove_incorrect_words_from_df(df, column, incorrect_words):
    """
    remove_incorrect_words_from_df removes all incorrect words from a specified column in a DataFrame.

    :param df: Input DataFrame
    :param column: Column in the DataFrame from which incorrect words are to be removed
    :param incorrect_words: List of incorrect words to be removed
    :return: DataFrame with incorrect words removed from the specified column
    """

    df[column] = df[column].apply(split_into_words)
    df[column] = df[column].apply(lambda x: filter_out_words(x, incorrect_words))
    df[column] = df[column].apply(join_words)
    return df

def remove_nonword_chars(word_list):
    """
    remove_nonword_chars takes a list of words and returns a new list where 
    all non-word characters have been removed from each word.

    :param word_list: List of words to be cleaned
    :return: New list of cleaned words
    """

    return [pd.Series(element).str.replace('[^\w\s]', '', regex=True).item() for element in word_list]

def get_unique_words(df, column):
    """
    get_unique_words returns all unique words from a specific column in a DataFrame.

    :param df: Input DataFrame
    :param column: Column name to process
    :return: Numpy array of unique words
    """

    unique_words = np.unique([item for sublist in df[column] for item in sublist])
    return unique_words

def count_category_frequencies(df, column='categories'):
    """
    count_category_frequencies converts a DataFrame column of lists represented as strings into actual lists,
    counts the frequency of each unique string across all lists, and returns a DataFrame summarizing
    the frequencies.

    :param df: The input DataFrame
    :param column: The name of the column to be processed
    :return: DataFrame summarizing the frequencies of each unique string
    """

    # Convert string to list
    df[column] = df[column].apply(ast.literal_eval)

    # Create an empty list
    all_strings = []

    # Iterate over the DataFrame
    for _, row in df.iterrows():
        # Extend all_strings with the strings in the current row
        all_strings.extend(row[column])

    # Count the frequencies of each unique string
    counted_data = Counter(all_strings)

    # Create a DataFrame from the Counter object
    summary_df = pd.DataFrame.from_records(list(counted_data.items()), columns=['Category', 'Frequency'])

    return summary_df

#Pickle helpers

def load_pickle(dataset, path) -> DataFrame:
    """
    load_pickle loads a preprocessed DataFrame from a pickled file based on the provided dataset name.

    :param dataset: A string indicating the name of the dataset, it should be either "UPV" or "Wikipedia".
    :param path: A string indicating the path where the pickled files are stored.
    :return: A pandas DataFrame loaded from the pickled file.

    The function raises a ValueError if the provided dataset name is not "UPV" or "Wikipedia".
    """

    if dataset == "UPV":
        preprocessed_data = read_pickle('mediaUPV-preprocessed.pickle', path)
        titles = preprocessed_data.get('Title', [])
        transcriptions = preprocessed_data.get('Transcription', [])
        keywords = preprocessed_data.get('Keywords', [])
        
        df = pd.DataFrame(titles, columns=['Title'])
        df['Transcription'] = transcriptions
        df['Keywords'] = keywords
        return df
        
    elif dataset == "Wikipedia":
        preprocessed_data = read_pickle('wikipedia-preprocessed.pickle', path)
        titles = preprocessed_data.get('title', [])
        texts = preprocessed_data.get('text', [])
        categories = preprocessed_data.get('categories', [])
        
        df = pd.DataFrame(titles, columns=['title'])
        df['text'] = texts
        df['categories'] = categories
        return df
    else:
        raise ValueError(f"Invalid dataste with name {dataset} in '{path}'.")

def read_pickle(name, path):
    """
    read_pickle reads and loads a pickled object from a file.

    :param name: The name of the pickle file (without the .pickle extension).
    :param path: The path where the pickle file is stored.
    :return: The loaded object.

    This function raises a ValueError if the specified pickle file does not exist.
    """

    if os.path.exists(f"{path}{name}.pickle"):
        with open(name, 'rb') as f:
            return pd.read_pickle(f)
    else:
        raise ValueError(f"Invalid pickle with name {name} in '{path}'.")

def save_pickle(data, name, path):
    """
    save_pickle saves an object to a pickled file.

    :param name: The name of the pickle file (without the .pickle extension) to be created.
    :param path: The path where the pickle file should be stored.
    :return: None.

    Note: This function overwrites any existing file with the same name in the specified path.
    """

    with open(f"{path}{name}.pickle", "wb") as file:
        pickle.dump(data, file)