import pandas as pd
from sklearn.decomposition import PCA
import logging 
import plotly.express as px
from dash import Dash, dcc, html 
from dash.dependencies import Input, Output 
import util
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# logger.info(f"To simulate my data store I am generating a tabular set with shape '{df.shape}'")
# logger.info(f"For those familiar with RAG you may recognize that as a common embedding cardinality")


def project_down(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=3)
    pca_arr = pca.fit_transform(df.drop(['fraud', 'id'], axis=1))
    pca_df = pd.DataFrame(pca_arr, columns=[f'PC{i+1}' for i in range(pca_arr.shape[1])])
    # logger.info(f"Next I perform PCA to reduce the dimensions down to '{pca_df.shape}' so we can visualize it")
    # logger.info(f"Attempting to render a 1536 dimensional point cloud would break my brain and set my GPU on fire")
    pca_df['fraud'] = df['fraud'].astype(str).values # to get the plotly coloring to work right it needs to be a string 
    pca_df['id'] = df['id'].values
    return pca_df


def create_dash_app(df: pd.DataFrame, reduced_df: pd.DataFrame, model) -> Dash:
    fig = px.scatter_3d(
        data_frame=reduced_df,
        x='PC1', y='PC2', z='PC3',
        color='fraud',
        title='PCA Point Cloud',
        custom_data=['id'],
        color_discrete_map={'0': 'blue', '1': 'red'}
    )

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='3d-scatter', figure=fig, style={'width': '100vw', 'height': '80vh'}),
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
        id_value = point['customdata'][0]

        rows = df[df['id'] == id_value]
        match len(rows):
            case 1: 
                label = rows['fraud'].values[0]
            case n if n > 1: 
                raise ValueError(f"There was somehow a duplicate id in the training population")
            case 0:
                raise ValueError(f"No row found with id: {id_value}")
            
        hover_row = df[df['id'] == id_value].drop(['fraud', 'id'], axis=1).values
        prediction = model.predict(hover_row)[0]
        answer = {
            0: 'not fraud',
            1: 'fraud'
        }
        return f"ID: {id_value}\nML Model Prediction: {answer[prediction]}\nTrue Label: {answer[label]}"

    return app

if __name__ == '__main__':
    df = util.load_data('fake_engineered_features.csv')
    pca_df = project_down(df)
    model = util.load_model('xgboost_model.pkl')
    app = create_dash_app(df, pca_df, model)
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)