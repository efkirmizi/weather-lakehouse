import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator

# Define the outlet dataset URI
raw_weather_dataset = Dataset("iceberg://nessie.weather.observations")

with DAG(
    dag_id="01_extract_weather_data",
    description="Extracts historical & daily weather observations into Iceberg",
    schedule="@daily",
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["etl", "spark", "iceberg"],
) as dag:

    pyspark_etl_task = DockerOperator(
        task_id="run_pyspark_etl",
        image="dag-pyspark-etl:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        environment={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        },
        command="python3 /opt/spark/work-dir/weather_etl.py",
        docker_url="unix://var/run/docker.sock",
        outlets=[raw_weather_dataset],
        retries=0,
        execution_timeout=timedelta(hours=2),
    )