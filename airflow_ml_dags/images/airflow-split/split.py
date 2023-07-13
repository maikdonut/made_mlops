import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split_data(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
    X_train.to_csv(os.path.join(output_dir, "x_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    split_data()