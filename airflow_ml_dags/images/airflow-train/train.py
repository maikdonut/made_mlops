import os
import click
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def train(data_dir: str, artifacts_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    model = LogisticRegression()
    X_train = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))

    path_scaler = os.path.join(artifacts_dir, "scaler.pkl")
    with open(path_scaler, "rb") as file:
        scaler = pickle.load(file)

    X_train_scaled = scaler.transform(X_train)
    model.fit(X_train_scaled, y_train)
    path = os.path.join(artifacts_dir, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
