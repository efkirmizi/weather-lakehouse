from datetime import datetime, timedelta
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator

from alerting import DEFAULT_ARGS
from credentials import s3_env

# Define the outlet dataset URI
raw_weather_dataset = Dataset("iceberg://nessie.weather.observations")

with DAG(
    dag_id="01_extract_weather_data",
    description="Extracts historical & daily weather observations into Iceberg",
    schedule="@daily",
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["etl", "spark", "iceberg"],
    default_args=DEFAULT_ARGS,
    is_paused_upon_creation=False,
) as dag:

    pyspark_etl_task = DockerOperator(
        task_id="run_pyspark_etl",
        image="dag-pyspark-etl:1.0",
        api_version="auto",
        auto_remove="success",
        network_mode="lakehouse-net",
        environment=s3_env(),
        command="python3 /opt/spark/work-dir/weather_etl.py",
        docker_url="unix://var/run/docker.sock",
        outlets=[raw_weather_dataset],
        retries=0,
        execution_timeout=timedelta(hours=2),
    )