from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import random


def train_model(dataset, model, embedding, saved_data=False):
    """
    train_model trains a model based on the specified model type.

    :param dataset: The dataset to use for training.
    :param model: The type of model to train ('LDA', 'K-Means', 'SVM', 'BERT')
    :param embedding: The type of embedding to use for data preprocessing ('CV', 'TF-IDF')
    :param saved_data: Indicates whether to use saved preprocessed data if available.
    :return: Trained model
    """
    
    # Preprocess the data based on the selected dataset
    df = preprocess_data(saved_data, dataset)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # Vectorize the training data
    X_train, vectorizer = choose_vectorizer(embedding, X_train)

    # Train selected model
    if model == 'LDA':
        trained_model = train_lda_model(X_train, n_components=6, random_state=42)
    elif model == 'K-Means':
        trained_model = train_kmeans_model(X_train, n_clusters=6, random_state=42)
    elif model == 'SVM':
        trained_model = train_svm_model(X_train, y_train, kernel='linear', random_state=42)
    elif model == 'BERT':
        if isinstance(X_train, str):
            trained_model = train_bert_seq2seq_model(df, model, tokenizer)
        else:
            raise ValueError("Invalid data type for BERT model. Expected raw text data.")
    else:
        raise ValueError("Invalid model type. Expected one of: 'LDA', 'K-Means', 'SVM', 'BERT'")
    
    # Save model and vectorizer for future use
    save_model(trained_model, model)
    save_vectorizer(vectorizer, embedding)
    
    return trained_model

def train_lda_model(dtm, n_components=3, random_state=42):
    """
    train_lda_model trains an LDA model on the provided document-term matrix.

    :param dtm: Document-Term matrix
    :param n_components: The number of topics (default: 3)
    :param random_state: Random seed for reproducibility (default: 42)
    :return: Trained LDA model
    """
    
    lda = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    lda.fit(dtm)
    return lda

def train_kmeans_model(dtm, n_clusters=6, random_state=42):
    """
    train_kmeans_model trains a K-Means model on the provided document-term matrix and also prints the topics.

    :param dtm: Document-Term matrix
    :param n_clusters: The number of clusters (default: 6)
    :param random_state: Random seed for reproducibility (default: 42)
    :return: Trained K-Means model and topic-results.
    """

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1, random_state=random_state)
    kmeans.fit(dtm)

    topic_results = kmeans.transform(dtm)

    # Print topics
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(n_clusters):
        print(f"Topic {i}")
        print([terms[ind] for ind in order_centroids[i, :10]])
        print('\n')

    return kmeans, topic_results

def train_svm_model(X_train, y_train, kernel='linear', random_state=42):
    """
    train_svm_model trains an SVM model on the provided training data and labels.

    :param X_train: Training data
    :param y_train: Labels for training data
    :param kernel: Kernel type to be used in the algorithm (default: 'linear')
    :param random_state: Random seed for reproducibility (default: 42)
    :return: Trained SVM model
    """

    svm = SVC(kernel=kernel, random_state=random_state)
    svm.fit(X_train, y_train)
    return svm

def train_bert_seq2seq_model(df, model, tokenizer, batch_size=2):
    """
    train_bert_seq2seq_model trains a BERT sequence-to-sequence model.

    :param df: DataFrame containing the training data.
    :param model: Initialized BERT model.
    :param tokenizer: Initialized tokenizer.
    :param batch_size: Size of the batches for training.
    :return: Trained model
    """

    # Create a dataset from the DataFrame
    class TextToCategoriesDataset(Dataset):
        def __init__(self, df):
            self.dataset = df

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset.iloc[idx]['text']
            categories = self.dataset.iloc[idx]['categories']
            text_processed = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            categories_processed = tokenizer(categories, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

            batch = {}
            batch["input_ids"] = text_processed.input_ids[0]
            batch["attention_mask"] = text_processed.attention_mask[0]
            batch["decoder_input_ids"] = categories_processed.input_ids[0]
            batch["decoder_attention_mask"] = categories_processed.attention_mask[0]
            batch["labels"] = categories_processed.input_ids.clone()
            batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]][0]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.long)

            return batch

    # Initialize the Dataset and DataLoader
    ttcd = TextToCategoriesDataset(df)
    dataloader = DataLoader(ttcd, batch_size=batch_size)

    # Define training arguments
    training_args = TrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        fp16=False, 
        output_dir="./",
        logging_steps=2,
        save_steps=10,
        eval_steps=4,
        local_rank=-1
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ttcd,
        tokenizer=tokenizer
    )
    
    trainer.train()

    return model

def print_random_words(cv, n=10):
    """
    print_random_words prints n randomly selected words from a CountVectorizer's feature names.

    :param cv: CountVectorizer object
    :param n: The number of random words to print (default: 10)
    :return: None
    """

    feature_names = cv.get_feature_names_out()
    
    for _ in range(n):
        random_word_id = random.randint(0, len(feature_names) - 1)
        print(feature_names[random_word_id])

def extract_top_words_for_topics(model, cv, n_words=15):
    """
    extract_top_words_for_topics prints the top words for each topic in the model.

    :param model: Trained LDA model
    :param cv: CountVectorizer object
    :param n_words: The number of top words to print for each topic (default: 15)
    :return: None
    """

    for index, topic in enumerate(model.components_):
        print(f'THE TOP {n_words} WORDS FOR TOPIC #{index}')
        print([cv.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]])
        print('\n')

def assign_topics(df, model, dtm):
    """
    assign_topics assigns topics to the DataFrame based on the LDA model.

    :param df: DataFrame
    :param model: Trained model
    :param dtm: Document-Term matrix
    :return: DataFrame with assigned topics
    """

    topic_results = model.transform(dtm)
    df['Topic'] = topic_results.argmax(axis=1)
    return df

def load_initial_model():
    """
    load_initial_model loads a pre-trained BERT model and tokenizer from Hugging Face model hub.

    This function checks if a GPU is available for training. If a GPU is available, it sets the device to 'cuda', 
    else, it uses 'cpu'. It then loads a pre-trained BERT model and tokenizer from the specified checkpoint. 
    After loading, the model is moved to the specified device and saved locally.

    Note: Ensure the specified checkpoint exists in Hugging Face's model hub.
    
    :return: None
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
    tokenizer = BertTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

    model.save_pretrained(save_model_directory)
    tokenizer.save_pretrained(save_tokenizer_directory)