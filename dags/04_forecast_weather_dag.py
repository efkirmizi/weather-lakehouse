import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule

from alerting import DEFAULT_ARGS
from credentials import s3_env

# Listen for daily precomputed features from the ETL
ml_features_dataset = Dataset("iceberg://nessie.weather.ml_features")

with DAG(
    dag_id="04_batch_inference_pipeline",
    description="Scalable Multi-Model Weather Forecasting Pipeline",
    schedule=[ml_features_dataset],
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["ml", "inference", "pytorch", "multi-model", "iceberg"],
    default_args=DEFAULT_ARGS,
    is_paused_upon_creation=False,
) as dag:
    # 1. The Drift Monitor Task
    check_data_drift = DockerOperator(
        task_id="monitor_concept_drift",
        image="dag-pytorch-model-training:1.0",
        api_version="auto",
        auto_remove="success",
        network_mode="lakehouse-net",
        environment={
            **s3_env(),
            # Forwarded from the scheduler so the monitor can call the Airflow API
            # without a credential baked into the image.
            # Defaulted rather than None: DockerOperator renders a None value as a
            # bare, valueless env var. drift_monitor already refuses to run with empty
            # credentials and says so, which is a far clearer failure.
            "AIRFLOW_API_USER": os.getenv("AIRFLOW_API_USER", ""),
            "AIRFLOW_API_PASSWORD": os.getenv("AIRFLOW_API_PASSWORD", ""),
        },
        command="python drift_monitor.py",
        docker_url="unix://var/run/docker.sock",
        retries=0,
        execution_timeout=timedelta(minutes=20),
    )

    # 2. The Inference Task (Your existing block)
    run_batch_inference = DockerOperator(
        task_id="generate_multi_model_forecasts",
        image="dag-pytorch-model-training:1.0",
        api_version="auto",
        auto_remove="success",
        network_mode="lakehouse-net",
        environment={
            **s3_env(),
            "MODEL_PREFIX": "Weather_Forecaster_" # Automatically captures all current and future models
        },
        command="python batch_inference.py",
        docker_url="unix://var/run/docker.sock",
        # No GPU request and no single_gpu slot: inference is one 72x4 forward pass
        # per champion through the CPU ONNX Runtime. Holding a GPU slot here would
        # only block training.
        retries=1,
        execution_timeout=timedelta(minutes=15),
        # Monitoring must not gate serving: a drift-check hiccup should never cost
        # us the daily forecast.
        trigger_rule=TriggerRule.ALL_DONE,
    )
    
    # 3. Define the Execution Order
    check_data_drift >> run_batch_inference