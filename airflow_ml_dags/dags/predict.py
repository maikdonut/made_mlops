from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from datetime import datetime
from docker.types import Mount
from utils import DEFAULT_ARGS, VOLUME_PATH, RAW_DATA_PATH, PATH_TO_ARTIFACTS, PATH_TO_PREDICTS, TARGET, wait_for_file

with DAG(
        dag_id="predict",
        start_date=datetime(2022, 12, 5),
        schedule_interval="@daily",
        default_args=DEFAULT_ARGS,
        tags=['hw3 MLOPS']
) as dag:

    wait_data = PythonSensor(
        task_id="wait-data",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/raw/{{ ds }}/data.csv"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    wait_scaler = PythonSensor(
        task_id="wait-scaler",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/artifacts/{{ ds }}/scaler.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    wait_model = PythonSensor(
        task_id="wait-model",
        python_callable=wait_for_file,
        op_args=["/opt/airflow/data/artifacts/{{ ds }}/model.pkl"],
        timeout=6000,
        poke_interval=10,
        retries=100,
        mode="poke"
    )

    predict = DockerOperator(
        image="airflow-predict",
        command=f"--data-dir={RAW_DATA_PATH} --artifacts-dir={PATH_TO_ARTIFACTS} --output-dir={PATH_TO_PREDICTS}",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        mounts=[Mount(source=VOLUME_PATH, target=TARGET, type='bind')]
    )

    [wait_data, wait_scaler, wait_model] >> predict
