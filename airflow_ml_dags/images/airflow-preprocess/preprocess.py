import os
import pickle
import click
import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    path = os.path.join(input_dir, "x_train.csv")
    train_data = pd.read_csv(path)
    scaler = StandardScaler()
    scaler.fit(train_data)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    preprocess()