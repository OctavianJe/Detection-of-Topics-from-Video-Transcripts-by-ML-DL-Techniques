from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def choose_vectorizer(embedding, documents):
    """
    choose_vectorizer chooses the appropriate text vectorizer (CountVectorizer or TfidfVectorizer) 
    based on the provided `embedding` argument and applies it to the `documents`.

    :param embedding: The type of text vectorization to apply ('CV' or 'TF-IDF').
    :param documents: The text data to be vectorized.

    :return: The vectorized data and the fitted vectorizer object.

    :raises ValueError: If the `embedding` argument is not 'CV' or 'TF-IDF'.
    """

    if embedding == "CV":
        vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words=union_exclusive_stopwords())
    elif embedding == "TF-IDF":
        vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words=union_exclusive_stopwords())
    else:
        raise ValueError(f"Invalid {embedding}")

    X = vectorizer.fit_transform(documents)

    return X, vectorizer