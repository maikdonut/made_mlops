from airflow import DAG
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime
from docker.types import Mount
from utils import DEFAULT_ARGS, VOLUME_PATH, RAW_DATA_PATH, PATH_TO_PROCESSED, PATH_TO_ARTIFACTS, TARGET, wait_for_file

with DAG(
        dag_id="train",
        start_date=datetime(2022, 12, 5),
        schedule_interval="@weekly",
        default_args=DEFAULT_ARGS,
        tags=['hw3 MLOPS']
) as dag:

    wait_data = PythonSensor(
        task_id="wait-data",
        python_callable=wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    wait_target = PythonSensor(
        task_id="wait-target",
        python_callable=wait_for_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/target.csv'],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    split_data = DockerOperator(
        image="airflow-split",
        command=f"--input-dir={RAW_DATA_PATH} --output-dir={PATH_TO_PROCESSED}",
        network_mode="bridge",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input-dir={PATH_TO_PROCESSED} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-scaler",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--data-dir={PATH_TO_PROCESSED} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="airflow-train-model",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--data-dir={PATH_TO_PROCESSED} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_ARTIFACTS}",
        network_mode="bridge",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    [wait_data, wait_target] >> split_data >> preprocess >> train >> validate