import os
import click
import pickle
import pandas as pd


@click.command("predict")
@click.option("--data-dir")
@click.option("--artifacts-dir")
@click.option("--output-dir")
def predict(data_dir: str, artifacts_dir: str, output_dir: str):
    X = pd.read_csv(os.path.join(data_dir, "data.csv"))
    path = os.path.join(artifacts_dir, "scaler.pkl")
    with open(path, "rb") as file:
        scaler = pickle.load(file)

    X_scaled = scaler.transform(X)

    path = os.path.join(artifacts_dir, "model.pkl")
    with open(path, "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X_scaled)
    os.makedirs(output_dir, exist_ok=True)
    preds = pd.DataFrame(y_pred, columns=['target'])
    preds.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    predict()