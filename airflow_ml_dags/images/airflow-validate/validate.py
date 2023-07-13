import os
import click
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


@click.command("validate")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def validate(data_dir: str, artifacts_dir: str, output_dir: str):
    X_test = pd.read_csv(os.path.join(data_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    path_scaler = os.path.join(artifacts_dir, "scaler.pkl")
    with open(path_scaler, "rb") as file:
        scaler = pickle.load(file)
    X_test_scaled = scaler.transform(X_test)

    path_model = os.path.join(artifacts_dir, "model.pkl")
    with open(path_model, "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test_scaled)
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_pred, y_pred)

    with open(os.path.join(output_dir, 'metric.json'), 'w') as file:
        json.dump(metrics, file)


if __name__ == "__main__":
    validate()