import numpy  as np
import pandas as pd
import yaml
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple

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



def read_data(data_path: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch the train and test data from the specified paths.

    Args:
        data_path: Dictionary containing file paths for train and test data.

    Returns:
        Dictionary of train and test DataFrames.
    """
    try:
        data = {
            'train_data': pd.read_csv(data_path['train']),
            'test_data': pd.read_csv(data_path['test'])
        }
        data['train_data'].fillna('', inplace=True)
        data['test_data'].fillna('', inplace=True)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        raise

def apply_tfidf(data: Dict[str, pd.DataFrame], params: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply tfidf transformation on the train and test datasets.

    Args:
        data: Dictionary containing train and test DataFrames.
        params: Dictionary containing parameters for CountVectorizer.

    Returns:
        Tuple of processed train and test DataFrames.
    """
    try:
        X_train = data['train_data']['content'].values
        y_train = data['train_data']['sentiment'].values

        X_test = data['test_data']['content'].values
        y_test = data['test_data']['sentiment'].values

        vectorizer = TfidfVectorizer(max_features=params['max_features'])
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_tfidf.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray())
        test_df['label'] = y_test

        logger.info("tfidf transformation applied successfully.")
        return train_df, test_df
    except KeyError as e:
        logger.error(f"Key error in tfidf parameters: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during tfidf of Words transformation: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Save processed train and test datasets into the specified folder.

    Args:
        train_df: Processed training dataset as a DataFrame.
        test_df: Processed testing dataset as a DataFrame.
    """
    try:
        data_path = os.path.join("data", "features")
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_tfidf.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_tfidf.csv"), index=False)

        logger.info(f"Processed data saved successfully in {data_path}.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def main() -> None:
    """
    Main function to execute the feature engineering pipeline.
    """
    try:
        data_path = {
            'train':  r'D:\mlops_cc1\emotion_detection\data\raw\train.csv',
            'test': r'D:\mlops_cc1\emotion_detection\data\raw\test.csv'
        }
        params_path = "params.yaml"

        # Load configuration parameters
        params = yaml.safe_load(open(params_path, 'r'))['feature_engineering']
        logging.info("Configuration parameters loaded successfully.")

        # Load and process data
        data = read_data(data_path)
        train_df, test_df = apply_tfidf(data, params)

        # Save the processed data
        save_data(train_df, test_df)

        logger.info("Feature engineering pipeline completed successfully.")
    except FileNotFoundError as e:
        logger.error(f"File not found during pipeline execution: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during pipeline execution: {e}")
        raise

if __name__ == '__main__':
    main()
