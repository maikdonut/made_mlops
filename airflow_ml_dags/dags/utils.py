import os
from datetime import timedelta


DEFAULT_ARGS = {
    "owner": "admin",
    "email": ["admin@example.com"],
    'email_on_failure': True,
    'email_on_retry': True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

VOLUME_PATH = "/home/michael/airflow_ml_dags/data"
TARGET = "/data"
RAW_DATA_PATH = "/data/raw/{{ ds }}"
PATH_TO_PROCESSED = "/data/processed/{{ ds }}"
PATH_TO_ARTIFACTS = "/data/artifacts/{{ ds }}"
PATH_TO_PREDICTS = "/data/predictions/{{ ds }}"


def wait_for_file(file_name):
    return os.path.exists(file_name)
