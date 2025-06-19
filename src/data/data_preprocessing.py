import numpy  as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK packages
nltk.download('wordnet')
nltk.download('stopwords')

# Creating logger
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Creating console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

#creating formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(message)s')


#now connecting all three together
#1 . console_handler to formatter
console_handler.setFormatter(formatter)
# logger to consloe_handler
logger.addHandler(console_handler)


#for filehandler 
file_handler=logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def load_data(train_test_data: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    try:
        data = {
            'train_data': pd.read_csv(train_test_data['train_data']),
            'test_data': pd.read_csv(train_test_data['test_data'])
        }
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

# Function to perform lemmatization
def lemmatization(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    try:
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        logger.debug("Lemmatization performed.")
        return text
    
    except Exception as e:
        logger.error(f"Error in lemmatization: {e}")
        raise

# Function to remove stop words
def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    try:
        text = [word for word in str(text).split() if word not in stop_words]
        logger.debug("Stop words removed.")
        return text
    except Exception as e:
        logger.error(f"Error removing stop words: {e}")
        raise

# Function to remove numbers
def removing_numbers(text: str) -> str:
    try:
        text = ''.join([char for char in text if not char.isdigit()])
        logger.debug("Numbers removed.")
        return text
    except Exception as e:
        logger.error(f"Error removing numbers: {e}")
        raise

# Function to convert text to lowercase
def lower_case(text: str) -> str:
    try:
        text = text.split()
        text = [word.lower() for word in text]
        logger.debug("Lowercase conversion performed.")
        return text
    except Exception as e:
        logger.error(f"Error converting to lowercase: {e}")
        raise

# Function to remove punctuations
def removing_punctuations(text: str) -> str:
    try:
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\\s+', ' ', text).strip()
        logger.debug("Punctuations removed.")
        return text
    except Exception as e:
        logger.error(f"Error removing punctuations: {e}")
        raise

# Function to remove URLs
def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        logger.debug("URLs removed.")
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"Error removing URLs: {e}")
        raise

# Function to normalize text in a dataframe
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        logger.info("Text normalization completed.")
        return df
    except Exception as e:
        logger.error(f"Error normalizing text: {e}")
        raise

def processed_data(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_processed_data = normalize_text(data['train_data'])
        test_processed_data = normalize_text(data['test_data'])
        logger.info("Data processed successfully.")
        return train_processed_data, test_processed_data
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def mkdir(train_processed_data: pd.DataFrame, test_processed_data: pd.DataFrame) -> None:
    try:
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logger.info("Processed data stored successfully.")
    except Exception as e:
        logger.error(f"Error creating directory or saving files: {e}")
        raise

def main() -> None:
    train_test_data = {
     'train_data':  r'D:\mlops_cc1\emotion_detection\data\raw\train.csv',
     'test_data':   r'D:\mlops_cc1\emotion_detection\data\raw\test.csv'
}


    
    try:
        data = load_data(train_test_data)
        train_processed_data, test_processed_data = processed_data(data)
        mkdir(train_processed_data, test_processed_data)
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
