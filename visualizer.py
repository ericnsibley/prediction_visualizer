import pandas as pd
from sklearn.decomposition import PCA
import logging 
import plotly.express as px
from dash import Dash, dcc, html 
from dash.dependencies import Input, Output 
import xgboost as xgb
import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# logger.info(f"To simulate my data store I am generating a tabular set with shape '{df.shape}'")
# logger.info(f"For those familiar with RAG you may recognize that as a common embedding cardinality")


def project_down(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=3)
    pca_arr = pca.fit_transform(df.drop(['fraud', 'id'], axis=1))
    pca_df = pd.DataFrame(pca_arr, columns=[f'PC{i+1}' for i in range(pca_arr.shape[1])])
    logger.info(f"Next I perform PCA to reduce the dimensions down to '{pca_df.shape}' so we can visualize it")
    logger.info(f"Attempting to render a 1536 dimensional point cloud would break my brain and set my GPU on fire")
    pca_df['fraud'] = df['fraud'].astype(str).values # to get the plotly coloring to work right it needs to be a string 
    pca_df['id'] = df['id'].values
    return pca_df


def create_dash_app(df: pd.DataFrame, model) -> Dash:
    fig = px.scatter_3d(
        data_frame=df,
        x='PC1', y='PC2', z='PC3',
        color='fraud',
        title='PCA Point Cloud',
        custom_data=['id'],
        color_discrete_map={'0': 'blue', '1': 'red'}
    )

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='3d-scatter', figure=fig),
        html.Div(id='hover-data', style={'whiteSpace': 'pre-line'})
    ])

    @app.callback(
        Output('hover-data', 'children'),
        [Input('3d-scatter', 'hoverData')]
    )
    def display_hover_data(hoverData):
        if hoverData is None:
            return 'Hover over a point to see its prediction'

        point = hoverData['points'][0]
        logger.info(f"point: {point}")
        id_value = point['customdata'][0]

        original_data_point = df[df['id'] == id_value].drop(['fraud', 'id'], axis=1).values
        dmatrix = xgb.DMatrix(original_data_point)
        prediction = model.predict(dmatrix)[0]

        return f"ID: {id_value}\nPrediction: {prediction}"

    return app

if __name__ == '__main__':
    df = util.load_data('fake_engineered_features.csv')
    pca_df = project_down(df)
    model = util.load_model('xgboost_model.pkl')
    app = create_dash_app(pca_df, model)
    app.run_server(debug=True)