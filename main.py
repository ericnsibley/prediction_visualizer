import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
import logging 
import plotly.express as px
import plotly.io as pio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

row_count = 10000
col_count = 1536


def generate_data(rows: int, cols: int) -> pd.DataFrame: 
    data = np.random.rand(rows, cols)
    df = pd.DataFrame(data=data)
    df['fraud'] = np.random.choice([0, 1], size=rows, p=[0.9, 0.1]) # I could attach this col after the PCA, but I wanted to structure my data source like how I do at work
    logger.info(f"To simulate my data store I am generating a tabular set with shape '{df.shape}'")
    logger.info(f"For those familiar with RAG you may recognize that as a common embedding cardinality")
    return df 


def project_down(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=3)
    pca_arr = pca.fit_transform(df.drop('fraud', axis=1))
    pca_df = pd.DataFrame(pca_arr, columns=[f'PC{i+1}' for i in range(pca_arr.shape[1])])
    logger.info(f"Next I perform PCA to reduce the dimensions down to '{pca_df.shape}' so we can visualize it")
    logger.info(f"Attempting to render a 1536 dimensional point cloud would break my brain and set my GPU on fire")
    pca_df['fraud'] = df['fraud'].astype(str).values # to get the plotly coloring to work right it needs to be a string 
    return pca_df


if __name__ == "__main__":
    df = generate_data(row_count, col_count)
    pca_df = project_down(df)

    fig = px.scatter_3d(
        data_frame=pca_df, 
        x='PC1', y='PC2', z='PC3', 
        color='fraud', 
        title='PCA Point Cloud',
        color_discrete_map={'0': 'blue', '1': 'red'}
    )
    # fig.update_layout(scene={
    #     'xaxis_title'
    # })
    fig.show()