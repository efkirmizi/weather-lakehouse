from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from alerting import DEFAULT_ARGS
from credentials import s3_env

# Mirrors TABLES_TO_MAINTAIN in jobs/maintenance/lakehouse_janitor.py; kept here so the
# scheduler can fan out without importing the job image's code.
MAINTAINED_TABLES = [
    "weather.observations",
    "weather.ml_features",
    "weather.forecast_predictions",
    "weather.scaling_parameters",
]

with DAG(
    dag_id="05_lakehouse_maintenance",
    description="Compacts Iceberg data files and rewrites manifests to keep query planning cheap.",
    schedule="0 3 * * 0",  # Runs every Sunday at 3:00 AM
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["data-engineering", "iceberg", "pyspark", "maintenance"],
    default_args=DEFAULT_ARGS,
    is_paused_upon_creation=False,
) as dag:

    # One mapped task per table: a failure on forecast_predictions should not force
    # re-running compaction on observations, and the UI shows which table is slow.
    run_janitor = DockerOperator.partial(
        task_id="run_lakehouse_janitor",
        map_index_template="{{ task.environment['MAINTENANCE_TABLE'] }}",
        image="dag-lakehouse-janitor:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        # The script is baked into the image, like every other job here. A bind
        # mount would be resolved by the Docker daemon against the *host* filesystem,
        # not against the scheduler container that computes the path.
        command="python3 /opt/spark/work-dir/lakehouse_janitor.py",
        docker_url="unix://var/run/docker.sock",
        retries=1,
        execution_timeout=timedelta(hours=1), # Compaction can take time as data grows
    ).expand(
        environment=[
            {**s3_env(), "MAINTENANCE_TABLE": table}
            for table in MAINTAINED_TABLES
        ]
    )