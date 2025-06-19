import os
import numpy as np
import pandas as pd
import yaml
import pickle
import xgboost as xgb
import logging
from typing import Tuple

# Setup Logger
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_data(train_bow: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_bow)
        logger.info("Training data loaded successfully.")
        return train_data
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise


def split_data(train_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    try:
        X_train = train_data.iloc[:, :-1].values  # All columns except the last
        Y_train = train_data.iloc[:, -1].values   # Only the last column
        logger.info("Data split into features and labels successfully.")
        return X_train, Y_train
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def model_train(X_train: np.ndarray, Y_train: np.ndarray, params: dict) -> xgb.XGBClassifier:
    try:
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            eta=params['eta'],
            max_depth=params['max_depth'],
            subsample=0.8,
            colsample_bytree=0.8
        )
        trained_model = xgb_model.fit(X_train, Y_train)
        logger.info("Model trained successfully.")
        return trained_model
    except KeyError as e:
        logger.error(f"Missing model parameter: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


def save_model(model: xgb.XGBClassifier) -> None:
    try:
        model_dir = os.path.join("models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")

        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        logger.info(f"Model saved successfully at {model_path}.")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise


def main() -> None:
    try:
        # Automatically detect project base path
        base_path = os.getcwd()

        train_path = os.path.join(base_path, "data", "features", "train_tfidf.csv")
        params_path = os.path.join(base_path, "params.yaml")

        # Load data
        train_data = load_data(train_path)
        X_train, Y_train = split_data(train_data)

        # Load parameters
        with open(params_path, 'r') as file:
            all_params = yaml.safe_load(file)

        if 'model_building' not in all_params:
            raise KeyError("Missing 'model_building' section in params.yaml")

        params = all_params['model_building']
        logger.info("Model parameters loaded successfully from YAML.")

        # Train model
        trained_model = model_train(X_train, Y_train, params)

        # Save model
        save_model(trained_model)

        logger.info("Model building pipeline executed successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found during pipeline execution: {e}")
    except KeyError as e:
        logger.error(f"YAML config key error: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during model building: {e}")
        raise


if __name__ == '__main__':
    main()

