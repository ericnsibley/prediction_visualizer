import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#remote: error: File fake_engineered_features.csv is 282.34 MB; this exceeds GitHub's file size limit of 100.00 MB 
row_count = 1000
col_count = 500


def generate_data(rows: int, cols: int) -> pd.DataFrame: 
    data = np.random.rand(rows, cols)
    df = pd.DataFrame(data=data)
    df['fraud'] = np.random.choice([0, 1], size=rows, p=[0.9, 0.1]) 
    df['id'] = np.arange(len(df))
    return df 


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(['fraud', 'id'], axis=1)
    scaler = StandardScaler()
    normalized = pd.DataFrame(scaler.fit_transform(features))
    normalized['fraud'] = df['fraud']
    normalized['id'] = df['id']
    return normalized


if __name__ == "__main__":
    df = generate_data(row_count, col_count)
    df = normalize_data(df)
    outfile = 'fake_engineered_features.csv'
    df.to_csv(outfile)
    logger.info(f'Data with shape {df.shape} has been written to {outfile}')