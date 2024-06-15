import pandas as pd
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame):
    x = df.drop(['fraud', 'id'], axis=1)
    y = df['fraud']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb_tree = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_tree.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

    y_pred = xgb_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy}")
    return xgb_tree


if __name__ == "__main__":
    df = util.load_data('fake_engineered_features.csv')
    model = train_model(df)
    util.store_model(model, 'xgboost_model.pkl')
