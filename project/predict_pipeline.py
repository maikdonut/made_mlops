import click
import logging
import pickle
import pandas as pd
from src.data import read_data
from src.features import create_transformer
from src.models import predict_model
from src.enities import read_predict_pipeline_params, PredictPipelineParams

logger = logging.getLogger('Predict pipeline')
logger.setLevel(logging.INFO)


def predict_pipeline(params: PredictPipelineParams):
    logger.info(f"Starting predict pipeline with params {params}")
    data = read_data(params.path_to_data)
    logger.info(f"Data shape is {data.shape}")

    transformer = create_transformer(params.feature_params)
    X = transformer.fit_transform(data)
    logger.info(f"Shape of transformed data is {X.shape}")
    logger.info(f"Loading model from path: {params.path_to_model}")
    with open(params.path_to_model, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Successfully loaded model: {type(model).__name__}")
    predictions = predict_model(X, model)
    pd.Series(predictions).to_csv(params.path_to_predictions, index=False)
    logger.info(f"Prediction saved by {params.path_to_predictions}")


@click.command(name="predict_pipeline")
@click.argument("config_path", default='configs/predict_config.yaml',
                type=click.Path(exists=True))
def predict_pipeline_command(config_path: str):
    params = read_predict_pipeline_params(config_path)
    predict_pipeline(params)


if __name__ == '__main__':
    predict_pipeline_command()
