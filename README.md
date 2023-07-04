# Detection of Topics from Video Transcripts by ML&DL Techniques

Diploma project, UCv, July 2023


Repository overview:


- data:
    This folder contains both datasets (UPV and Wikipedia (from [https://huggingface.co/datasets/olm/wikipedia][olm/wikipedia])) and all the models trained to be used for evaluation.

- notebook:
    This folder contains the notebook used during the development process

- src:
    This folder contains all source code used for creating the application

- visuals:
      This folder contains all graphs and scores from the topic models.

Instructions for instalation:


1. Create virtual environment (this project runs on Python 3.10.11):
    conda create --name detection-of-topics python=3.10.11


3. Activate virtual environment:
    conda activate detection-of-topics


3. Fetch project requirements:
    pip install -r requirements.txt


4. Run main.py:
    python main.py --dataset DATASET --model SAVED_MODEL


Arguments:
- (--dataset) is used to select the dataset:
    UPV: UPV
    Wikipedia: Wikipedia
    input: input

- (--model) is used to set the topic model:
    Machine Learning (ML):
        Latent Dirichlet Allocation: LDA
        K-Means: K-Means
        Suport Vector Machine: SVM
    Deep Learning (DL):
        Bidirectional Encoder Representations from Transformers: BERT

- (--train) is used to train the model (only for UPV and Wikipedia)

- (--embedding) is used to define the embedding used when training only an ML model: 
    Count Vectorizer: CV
    Term Frequency - Inverse Document Frequency: TF-IDF

- (--download-dataset) is used to download the base Wikipedia dataset from Hugging Face

- (--download-categories) is used to download the categories of each Wikipedia dataset using data scrapping technique

- (--file) is used to read content for model evaluation from a text file instead of Terminal when dataset is input