import json
import logging
import click
from src.data import extract_target, read_data, split_train_test_data
from src.enities import TrainConfig, read_train_pipeline_params
from src.features import create_transformer, process_features
from src.models import evaluate_model, predict_model, save_model, train_model

logger = logging.getLogger('Training pipeline')
logger.setLevel(logging.INFO)


def train_pipeline(params: TrainConfig):
    logger.info(f"Starting train pipeline with params {params}")
    data = read_data(params.path_to_data)
    logger.info(f"Data shape is {data.shape}")
    X, y = extract_target(data, params.feature_params.target_column)
    X_train, X_val, y_train, y_val = split_train_test_data(X, y, params.splitting_params)
    logger.info(f"Size of train data is {X_train.shape[0]}, Size of validation data is {X_val.shape[0]}")
    transformer = create_transformer(params.feature_params)
    transformer.fit_transform(X_train)
    X_train = process_features(transformer, X_train)
    model = train_model(X_train, y_train, params.train_params)
    X_val = transformer.transform(X_val)
    predictions = predict_model(X_val, model)
    metrics = evaluate_model(predictions, y_val)
    logger.info(f"Model evaluation {metrics}")

    with open(params.path_to_metrics, "w") as f:
        json.dump(metrics, f)
        logger.info(f"Model metrics saved by {params.path_to_metrics}")
    save_model(model, params.path_to_model)
    logger.info(f"Model saved by {params.path_to_model}")
    return metrics


@click.command(name="train_pipeline")
@click.argument("config_path", default='configs/train_log_reg.yaml',
                type=click.Path(exists=True))
def train_pipeline_command(config_path: str):
    params = read_train_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == '__main__':
    train_pipeline_command()
