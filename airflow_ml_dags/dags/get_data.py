from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from datetime import datetime

from utils import RAW_DATA_PATH, DEFAULT_ARGS, VOLUME_PATH, TARGET

with DAG(
        dag_id="get_data",
        start_date=datetime(2022, 12, 5),
        schedule_interval="@daily",
        default_args=DEFAULT_ARGS,
        tags=['hw3 MLOPS']
) as dag:
    get_data = DockerOperator(
        image="airflow-download",
        command=RAW_DATA_PATH,
        task_id="docker-airflow-get-data",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    get_data
