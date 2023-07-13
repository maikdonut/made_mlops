import click
import pandas as pd
from dataprep.eda import create_report


@click.command()
@click.option(
    "--path_to_data",
    type=click.Path(exists=True),
    default="project/data/raw/heart_cleveland_upload.csv",
    help="Path to input data",
)
@click.option(
    "--path_to_report",
    type=click.Path(exists=False),
    default="project/reports/eda.html",
    help="Path to save eda report",
)
def create_eda(path_to_data: str, path_to_report: str):
    df = pd.read_csv(path_to_data)
    eda = create_report(df)
    eda.save(path_to_report)


if __name__ == "__main__":
    create_eda()
