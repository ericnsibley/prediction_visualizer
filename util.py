import pandas as pd
import logging
import pickle
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded with shape {df.shape}")
    return df


def store_model(model, outfile_name) -> None:
    with open(outfile_name, 'wb') as outfile: 
        pickle.dump(model, outfile)
    logger.info(f"Model saved as {outfile_name}")


def load_model(model_file: str) -> XGBClassifier:
    with open(model_file, 'rb') as infile: 
        model = pickle.load(infile)
    return model