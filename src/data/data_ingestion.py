import numpy as np
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml

# ------------------ Logging Setup --------------------
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler (saved in root: emotion_detection/error.log)
file_handler = logging.FileHandler('error.log')  # keep as is
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ------------------ Load Parameters --------------------
def load_params(params_path: str = "params.yaml") -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.info(f'Loaded test_size: {test_size}')
        return test_size
    except Exception as e:
        logger.error(f"Error loading params.yaml: {e}")
        raise

# ------------------ Read CSV Data ----------------------
def read_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.info(f'Data loaded from: {file_path}')
        return df
    except Exception as e:
        logger.error(f'Error reading data from {file_path}: {e}')
        raise

# ------------------ Process Data ----------------------
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'], inplace=True)
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        df['sentiment'] = df['sentiment'].map({'happiness': 1, 'sadness': 0})
        logger.info('Data cleaned and processed')
        return df
    except Exception as e:
        logger.error(f'Error processing data: {e}')
        raise

# ------------------ Save Data -------------------------
def save_data(output_dir: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logger.info(f'Train/Test data saved at: {output_dir}')
    except Exception as e:
        logger.error(f'Error saving data: {e}')
        raise

# ------------------ Main Pipeline ---------------------
def main():
    try:
        test_size = load_params()
        df = read_data('data/external/tweet_emotions.csv')  # Place CSV here
        processed_df = process_data(df)
        train_df, test_df = train_test_split(processed_df, test_size=test_size, random_state=42)
        save_data('data/raw', train_df, test_df)
        logger.info("✅ Data ingestion pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
