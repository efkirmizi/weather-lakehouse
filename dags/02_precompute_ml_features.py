from datetime import datetime
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator

from alerting import DEFAULT_ARGS
from credentials import s3_env

# Define dataset triggers and outlets
raw_weather_dataset = Dataset("iceberg://nessie.weather.observations")
ml_features_dataset = Dataset("iceberg://nessie.weather.ml_features")

with DAG(
    dag_id="02_precompute_ml_features",
    description="Precomputes normalized features with PySpark MLlib",
    schedule=[raw_weather_dataset],
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["ml", "spark", "feature-engineering"],
    default_args=DEFAULT_ARGS,
    is_paused_upon_creation=False,
) as dag:

    feature_engineering_task = DockerOperator(
        task_id="run_pyspark_feature_engineering",
        image="dag-pyspark-feature-engineering:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        environment={
            **s3_env(),
            # The job appends only new observations unless the normalization has
            # drifted. Pass conf {"FEATURE_REBUILD": "full"} to renormalize the whole
            # table on demand.
            "FEATURE_REBUILD": "{{ dag_run.conf.get('FEATURE_REBUILD', '') }}",
        },
        command="python3 /opt/spark/work-dir/feature_engineering.py",
        docker_url="unix://var/run/docker.sock",
        outlets=[ml_features_dataset],
        retries=0,
    )