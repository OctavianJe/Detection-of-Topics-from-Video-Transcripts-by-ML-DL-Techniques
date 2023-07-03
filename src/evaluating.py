from src.miscellaneous import *
from src.preprocessing import *
from src.variables import *

from transformers import BertTokenizerFast, EncoderDecoderModel
from transformers import AutoTokenizer, EncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch


def evaluate_model(dataset, model, input_data):
    """
    evaluate_model 

    :param dataset: 
    :param model: 
    """

    evaluation_dataset = None

    if dataset == "input":
        evaluation_dataset = input_data
    elif dataset == "UPV":
        if os.path.exists(f'{get_root_projet()}{upv_path}{upv_dataset_name}.csv'):
            evaluation_dataset = pd.read_csv(f'{get_root_projet()}{upv_path}{upv_dataset_name}.csv', encoding='utf-8')
            clean_dataset(evaluation_dataset)
        else:
            raise ValueError(f"File '{upv_dataset_name}' from '{get_root_projet()}{upv_path}' is missing.")
    elif dataset == "Wikipedia":
        if os.path.exists(f'{get_root_projet()}{wikipedia_path}{wikipedia_without_categories_dataset_name}.csv'):
            evaluation_dataset = pd.read_csv(f'{get_root_projet()}{wikipedia_path}{wikipedia_without_categories_dataset_name}.csv', encoding='utf-8')
            clean_dataset(evaluation_dataset)
        else:
            raise ValueError(f"File '{wikipedia_without_categories_dataset_name}' from '{get_root_projet()}{wikipedia_path}' is missing.")
    else:
        raise ValueError(f"Invalid {dataset} dataset.")

    if model == "BERT":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if os.path.exists(f"{get_root_projet()}{bert_path}{summarize_text}{tokenizer}"):
            summarize_text_tokenizer = AutoTokenizer.from_pretrained(f"{get_root_projet()}{bert_path}{summarize_text}{tokenizer}")
        else:
            raise ValueError(f"Folder '{summarize_text}{tokenizer}' from '{get_root_projet()}{bert_path}' is missing.")

        if os.path.exists(f"{get_root_projet()}{bert_path}{summarize_text}{model_folder}"):
            summarize_text_model = EncoderDecoderModel.from_pretrained(f"{get_root_projet()}{bert_path}{summarize_text}{model_folder}")
        else:
            raise ValueError(f"Folder '{summarize_text}{model_folder}' from '{get_root_projet()}{bert_path}' is missing.")

        if os.path.exists(f"{get_root_projet()}{bert_path}{predict_topic_folder}{tokenizer}"):
            predict_topic_tokenizer = AutoTokenizer.from_pretrained(f"{get_root_projet()}{bert_path}{predict_topic_folder}{tokenizer}")
        else:
            raise ValueError(f"Folder '{predict_topic_folder}{tokenizer}' from '{get_root_projet()}{bert_path}' is missing.")

        if os.path.exists(f"{get_root_projet()}{bert_path}{predict_topic_folder}{model_folder}"):
            predict_topic_model = AutoModelForSequenceClassification.from_pretrained(f"{get_root_projet()}{bert_path}{predict_topic_folder}{model_folder}")
        else:
            raise ValueError(f"Folder '{predict_topic_folder}{model_folder}' from '{get_root_projet()}{bert_path}' is missing.")

        if isinstance(evaluation_dataset, str):
            summary = generate_summary(evaluation_dataset, summarize_text_tokenizer, summarize_text_model, device)
            print(summary)
            prediction = process_summary(summary, predict_topic_tokenizer, predict_topic_model)
            print(prediction)
        elif isinstance(evaluation_dataset, pd.DataFrame):
            if dataset == "UPV":
                evaluation_dataset['Topic'] = evaluation_dataset.apply(lambda row: process_row(row, summarize_text_tokenizer, summarize_text_model, predict_topic_tokenizer, predict_topic_model, device, 'Transcription'), axis=1)
                evaluation_dataset.to_csv(f'{get_root_projet()}{upv_path}{upv_dataset_name}-evaluation-{model}.csv', index=False)
            else:
                evaluation_dataset['topic'] = evaluation_dataset.apply(lambda row: process_row(row, summarize_text_tokenizer, summarize_text_model, predict_topic_tokenizer, predict_topic_model, device, 'text'), axis=1)
                evaluation_dataset.to_csv(f'{get_root_projet()}{wikipedia_path}{wikipedia_without_categories_dataset_name}-evaluation-{model}.csv', index=False)
        else:
            print(f"Unsupported {type(evaluation_dataset)} type.")
    elif model_type in ['LDA', 'SVM']:
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Report: \n{classification_report(y_test, y_pred)}")
    elif model_type == 'K-Means':
        y_pred = model.predict(X_test)
        print(f"Predicted labels: {y_pred}")
    else:
        raise ValueError("Invalid model type. Expected one of: 'LDA', 'K-Means', 'SVM', 'BERT'")

#BERT Helper

def process_row(row, summarize_text_tokenizer, summarize_text_model, predict_topic_tokenizer, predict_topic_model, device, column):
    """
    process_row generates a summary for a given text and predicts its topic.

    :param row: A row from a pandas DataFrame that contains the text data in the specified column.
    :param summarize_text_tokenizer: The tokenizer used to encode the text for the summarization model.
    :param summarize_text_model: The model used to generate the summary of the text.
    :param predict_topic_tokenizer: The tokenizer used to encode the summary for the topic prediction model.
    :param predict_topic_model: The model used to predict the topic of the summary.
    :param device: The device (CPU or GPU) that PyTorch uses for computations.
    :param column: The name of the column in the DataFrame that contains the text data.
    :return: The predicted topic of the summary of the given text.
    """
    text = row[column]
    summary = generate_summary(text, summarize_text_tokenizer, summarize_text_model, device)
    prediction = process_summary(summary, predict_topic_tokenizer, predict_topic_model)
    return prediction

def process_summary(summary, predict_topic_tokenizer, predict_topic_model):
    """
    process_summary is used to get the predicted topic of the given summary.
    
    :param summary: The input summary for which the topic is to be predicted.
    :param predict_topic_tokenizer: The tokenizer used for encoding the text.
    :param predict_topic_model: The model used for topic prediction.
    :return: The predicted topic of the given summary.
    """

    prediction = predict_topic(summary, predict_topic_tokenizer, predict_topic_model)
    return prediction

def generate_summary(text, predict_topic_tokenizer, predict_topic_model, device):
    """
    generate_summary is used to generate a summary for the given text.

    :param text: The input text for which a summary is to be generated.
    :param predict_topic_tokenizer: The tokenizer used for encoding the text.
    :param predict_topic_model: The model used for generating the summary.
    :param device: The device used for computations (CPU/GPU).
    :return: The generated summary for the given text.
    """

    inputs = predict_topic_tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = predict_topic_model.generate(input_ids, attention_mask=attention_mask)
    return predict_topic_tokenizer.decode(output[0], skip_special_tokens=True)

def predict_topic(text, predict_topic_tokenizer, predict_topic_model):
    """
    predict_topic is used to predict the topic for the given text.

    :param text: The input text for which the topic is to be predicted.
    :param predict_topic_tokenizer: The tokenizer used for encoding the text.
    :param predict_topic_model: The model used for topic prediction.
    :return: The predicted topic for the given text.
    """

    inputs = predict_topic_tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = predict_topic_model(**inputs)
    predictions = outputs.logits
    predicted_topic = torch.argmax(predictions).item()
    return predict_topic_model.config.id2label[predicted_topic]

def predict_topic_explicit(text, predict_topic_tokenizer, predict_topic_model):
    """
    predict_topic_explicit is used to predict the probabilities for each topic for the given text.

    :param text: The input text for which the probabilities are to be predicted.
    :param predict_topic_tokenizer: The tokenizer used for encoding the text.
    :param predict_topic_model: The model used for predicting the probabilities.
    :return: A dictionary containing each topic and its corresponding predicted probability.
    """

    inputs = predict_topic_tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = predict_topic_model(**inputs)
    predictions = outputs.logits
    probabilities = torch.nn.functional.softmax(predictions, dim=1)
    probabilities = probabilities[0].tolist()
    return {predict_topic_model.config.id2label[i]: prob for i, prob in enumerate(probabilities)}