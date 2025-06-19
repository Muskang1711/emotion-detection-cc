import os
import numpy as np
import pandas as pd
import pickle
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from typing import Tuple, Dict

# Logger setup
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler('error.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def load_model(model_path: str) -> object:
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info("Model loaded successfully.")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def load_test_data(test_path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        df = pd.read_csv(test_path)
        X_test = df.iloc[:, :-1].values
        Y_test = df.iloc[:, -1].values
        logger.info("Test data loaded and split successfully.")
        return X_test, Y_test
    except FileNotFoundError as e:
        logger.error(f"Test data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise


def evaluate_model(model: object, X_test: np.ndarray, Y_test: np.ndarray) -> Dict[str, float]:
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results = {
            "accuracy": accuracy_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(Y_test, y_pred, average='weighted', zero_division=0),
            "auc": roc_auc_score(Y_test, y_proba, multi_class='ovr') if len(set(Y_test)) > 2 else roc_auc_score(Y_test, y_proba)
        }

        logger.info("Evaluation metrics calculated successfully.")
        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


def save_metrics(metrics: Dict[str, float], output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


def main() -> None:
    try:
        base_path = os.getcwd()
        model_path = os.path.join(base_path, 'models', 'model.pkl')
        test_data_path = os.path.join(base_path, 'data', 'features', 'test_tfidf.csv')
        metrics_path = os.path.join(base_path, 'reports', 'metrics.json')

        model = load_model(model_path)
        X_test, Y_test = load_test_data(test_data_path)
        metrics = evaluate_model(model, X_test, Y_test)
        save_metrics(metrics, metrics_path)

        logger.info("Model evaluation pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Unexpected error during evaluation pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
