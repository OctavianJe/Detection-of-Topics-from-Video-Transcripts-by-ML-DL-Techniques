from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_wordcloud(words):
    """
    create_wordcloud creates a wordcloud from a given list of words.

    :param words: List of words
    :return: WordCloud object
    """

    wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate(' '.join(words))
    return wordcloud

def plot_wordcloud(wordcloud, title):
    """
    plot_wordcloud plots a given wordcloud.

    :param wordcloud: WordCloud object
    :param title: Title of the wordcloud plot
    :return: None
    """

    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout(pad = 0)
    plt.show()

def plot_word_distribution(df, column):
    """
    plot_word_distribution plots the distribution of words in a specific column of a DataFrame.

    :param df: Input DataFrame
    :param column: Column name to process
    :return: None
    """

    words = [item for sublist in df[column] for item in sublist]
    words_series = pd.Series(words)
    word_counts = words_series.value_counts()
    plt.figure(figsize=(10, 6))
    word_counts.plot(kind='bar')
    plt.title('Distribution of Keywords')
    plt.xlabel('Keyword')
    plt.ylabel('Frequency')
    plt.show()

def plot_pca(topic_results, model):
    """
    plot_pca takes the topic-results from a model and plots them using PCA.

    :param topic_results: The results of a model.
    :param model: Fitted model.
    :return: None
    """
    
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(topic_results)

    num_clusters = model.cluster_centers_.shape[0]
    cluster_colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    for cluster in range(num_clusters):
        plt.scatter(
            reduced_features[np.argmax(topic_results, axis=1) == cluster, 0],
            reduced_features[np.argmax(topic_results, axis=1) == cluster, 1],
            color=cluster_colors[cluster],
            label=f"Cluster {cluster}"
        )

    # Calculate the cluster centers
    cluster_centers = np.zeros((num_clusters, reduced_features.shape[1]))
    for cluster in range(num_clusters):
        cluster_centers[cluster] = np.mean(reduced_features[np.argmax(topic_results, axis=1) == cluster], axis=0)

    # Plot x markers for cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=150, c='b')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Cluster Distribution (PCA)')
    plt.legend()
    plt.show()

def plot_elbow_method(dtm, n_clusters=range(2, 11)):
    """
    plot_elbow_method performs the Elbow Method on the document-term matrix and plots the results.

    :param dtm: Document-Term Matrix.
    :param n_clusters: Range of clusters to try. Default is range(2, 11).
    :return: None
    """

    wcss = []
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(dtm)
        wcss.append(kmeans.inertia_)

    plt.plot(n_clusters, wcss, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method')
    plt.show()