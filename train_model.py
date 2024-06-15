import pandas as pd
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded with shape {df.shape}")
    return df


def train_model(df: pd.DataFrame):
    x = df.drop('fraud', axis=1)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb_tree = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_tree.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred = xgb_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy}")
    return xgb_tree


def store_model(model, outfile_name):
    with open(outfile_name, 'wb') as outfile: 
        pickle.dump(model, outfile)
    logger.info(f"Model saved as {outfile_name}")


if __name__ == "__main__":
    df = load_data('fake_engineered_features.csv')
    model = train_model(df)
    store_model(model, 'xgboost_model.pkl')
