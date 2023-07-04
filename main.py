from src.data import *
from src.evaluating import evaluate_model
from src.training import train_model
from src.variables import *

import argparse


def process_data_model(
    dataset, 
    model, 
    train, 
    download_dataset=None, 
    download_categories=None,
    embedding=None,
    file=None
):
    """
    process_data_model is used to handle the complete workflow of downloading, saving and processing a
    dataset with a specified machine learning model.

    :param dataset: Name of the dataset to be used, options are 'UPV', 'Wikipedia' or 'input'.
    :param model: Name of the model to be used, options are 'LDA', 'K-Means', 'SVM', or 'BERT'.
    :param train: A boolean flag to indicate whether the model needs to be trained.
    :param download_dataset: A boolean flag to indicate if the Wikipedia dataset needs to be 
                             downloaded, default is None.
    :param download_categories: A boolean flag to indicate if the Wikipedia article categories
                                need to be downloaded, default is None.
    :param embedding: Name of the embedding to be used for training, options are 'CV' (CountVectorizer) 
                      or 'TF-IDF' (Term Frequency-Inverse Document Frequency), default is None.
    :param file: Specify if data need to be read from file or from terminal
    """

    # Check that dataset is one of the allowed types
    if dataset not in ["UPV", "Wikipedia", "input"]:
        raise ValueError(f"Invalid {dataset} dataset.")

    # If dataset is Wikipedia, download it and categories if required
    if dataset == "Wikipedia":
        if download_dataset:
            wikipedia = download_wikipedia_dataset()
        
            if download_categories:
                save_wikipedia_with_categories_dataset(wikipedia)
            else:
                save_wikipedia_without_categories_dataset(wikipedia)

    input_data = None
    if dataset == "input":
        if file:
            # Insert data from text file
            with open(f"{get_root_projet()}{input_path}input.txt", "r") as file:
                input_data = file.read()
        else:
            # Insert data from terminal
            input_data = input("Enter the input: ")

    # Check that model is one of the allowed types
    if model not in ["LDA", "K-Means", "SVM", "BERT"]:
        raise ValueError(f"Invalid {model} model on {dataset} dataset.")

    # Process the data depending on the model and dataset
    if train:
        if dataset != "input":
            train_model(dataset, model, embedding)
        else:
            raise ValueError(f"Training not possible on {dataset} dataset.")
    else:
        evaluate_model(dataset, model, input_data)

def parse_arguments():
    """
    parse_arguments parses command line arguments required for the model. The required arguments are 'dataset' and 'model'.
    It also takes optional arguments for 'train', 'download_dataset', 'download_categories' and 'embedding' 
    depending on the dataset and model requirements.
    
    The 'dataset' can be one of the following: 'UPV', 'Wikipedia', or 'input'. If 'dataset' is not 'input', 
    then 'train' and 'embedding' arguments are also parsed.
    
    If the 'dataset' is 'Wikipedia', it parses two additional arguments 'download_dataset' and 'download_categories' 
    which specify whether the Wikipedia dataset and article categories need to be downloaded respectively.
    
    :return: args, the parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        type=str,
        required=True,
        help="The dataset that should be used: UPV (UPV), Wikipedia (Wikipedia), input"
    )
    parser.add_argument(
        '--model',
        dest='model',
        type=str,
        required=True,
        help="The model that should be used: LDA (LDA), K-Means (K-Means), SVM (SVM), BERT (BERT)"
    )

    args, unknown = parser.parse_known_args()
    assert args.dataset in ["UPV", "Wikipedia", "input"], (
        "Which dataset you want to use? Choose between 'UPV', 'Wikipedia' or 'input'.")
    assert args.model in ['LDA', 'K-Means', 'SVM', 'BERT'], (
        "Which model should be used? Choose between 'LDA', 'K-Means', 'SVM' or 'BERT'.")

    if args.dataset != 'input':
        parser.add_argument(
            '--train',
            dest='train',
            required=False,
            default=False,
            action='store_true',
            help="Do the model need to be trained?"
        )
        parser.add_argument(
            '--embedding',
            dest='embedding',
            type=str,
            required=False,
            help="The embedding that should be used: CV (CV), TF-IDF (TF-IDF)"
        )

        if args.dataset == 'Wikipedia':
            parser.add_argument(
                '--download-dataset',
                dest='download_dataset',
                required=False,
                default=False,
                action='store_true',
                help="Do the Wikipedia dataset need to be downloaded?"
            )
            parser.add_argument(
                '--download-categories',
                dest='download_categories',
                required=False,
                default=False,
                action='store_true',
                help="Do the Wikipedia article categories need to be downloaded?"
            )
    else:
        parser.add_argument(
            '--file',
            dest='file',
            required=False,
            default=False,
            action='store_true',
            help="Do the input data need to be read from file?"
        )

    args = parser.parse_args()

    return args

def main():
    """
    main is the main function that is executed when the script is run directly.

    It first calls the 'parse_arguments' function to parse the command-line arguments. 
    Then, it extracts these arguments and their values into 'args_dict'.

    It then passes these arguments to the 'process_data_model' function.

    The 'dataset' and 'model' arguments are required, whereas 'train', 'download_dataset', 
    'download_categories', and 'embedding' are optional arguments.
    """

    args = parse_arguments()
    args_dict = vars(args)
    process_data_model(
        args.dataset,
        args.model,
        args_dict.get('train', None),
        args_dict.get('download_dataset', None),
        args_dict.get('download_categories', None),
        args_dict.get('embedding', None),
        args_dict.get('file', None)
    )

if __name__ == "__main__":
    main()