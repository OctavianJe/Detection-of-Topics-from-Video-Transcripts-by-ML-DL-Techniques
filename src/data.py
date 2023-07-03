from src.miscellaneous import *
from src.variables import *

from datasets import load_dataset
from pandas import DataFrame

import json
import requests
import tqdm


def get_article_categories(url):
    """
    get_article_categories is used to aquire for each article the corresponding categories by using data scrapping technique and Wikipedia API

    :param url: Wikipedia dataset url column
    :return: a list of categories for each article in a list of string type
    """

    try:
        title = url.split('/')[-1]
        api_url = f"https://es.wikipedia.org/w/api.php?action=query&prop=categories&format=json&titles={title}"
        response = requests.get(api_url)

        if response.status_code == 200:
            print(response)
            data = response.json()

            if 'query' in data and 'pages' in data['query']:
                page_id = list(data['query']['pages'].keys())[0]
                categories = data['query']['pages'][page_id].get('categories', [])

                return [cat['title'].replace('Categoría:', '') for cat in categories if not 'hidden' in cat and not cat['title'].startswith('Categoría:Wikipedia:')]
            else:
                print(f"Unexpected response structure for URL {url}: {data}")
                return []
        else: 
            print(f"Unsuccessful request for URL {url}: status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error processing URL {url}: {str(e)}")
        return []

def download_wikipedia_dataset(
    name="olm/wikipedia", 
    language="es", 
    date="20230301"
) -> DataFrame:
    """
    download_wikipedia_dataset is used to download from Hugging Face the Wikipedia dataset without categories

    :param name: name of database
    :param language: language of database
    :param date: date when database is avalable
    :return: downloaded dataset in a DataFrame object type
    """

    dataset = load_dataset(name, language=language, date=date)
    # The dataset only provides training data
    return dataset['train'].to_pandas()

def save_wikipedia_without_categories_dataset(df):
    """
    save_wikipedia_without_categories_dataset is used to save in .csv format the downloaded dataset

    :param df: dataset to be saved in a DataFrame object type
    """

    df.to_csv(f'{get_root_projet()}{wikipedia_path}{wikipedia_without_categories_dataset_name}.csv', mode='w', encoding='utf-8')
    
def save_wikipedia_with_categories_dataset(df):
    """
    save_wikipedia_with_categories_dataset is used to save in .csv format the downloaded dataset adding also the categories of each article

    :param df: dataset to be saved in a DataFrame object type
    """

    for i in tqdm(range(len(df))):
        df.loc[i, 'categories'] = df.loc[i, 'url'].apply(get_article_categories)

        if i == 0:
            df.loc[i].to_csv(f"{get_root_projet()}{wikipedia_path}{wikipedia_with_categories_dataset_name}.csv", mode='w', encoding='utf-8')
        else:
            df.loc[i].to_csv(f"{get_root_projet()}{wikipedia_path}{wikipedia_with_categories_dataset_name}.csv", mode='a', encoding='utf-8', header=False)