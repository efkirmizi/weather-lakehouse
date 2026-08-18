import os
import requests
from datetime import datetime, timedelta
from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest

MODEL_NAME = "Weather_Forecaster_Transformer"

def _check_registry_status():
    """Pings MLflow API to see if the model exists. Routes to appropriate task."""
    url = f"http://mlflow:5000/api/2.0/mlflow/registered-models/get?name={MODEL_NAME}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"Model {MODEL_NAME} found in registry! Taking incremental path.")
            return "run_incremental_training"
        else:
            print(f"Model {MODEL_NAME} not found (Status {response.status_code}). Taking full training path.")
            return "run_full_training"
    except Exception as e:
        print(f"Failed to reach MLflow ({e}). Defaulting to full training.")
        return "run_full_training"

with DAG(
    dag_id="03b_train_pytorch_transformer",
    description="Intelligent weekly training for Transformer",
    schedule="0 2 * * 0",
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["ml", "pytorch", "training", "gpu", "attention", "smart-routing"],
    is_paused_upon_creation=False,
) as dag:

    # 1. The Routing Task
    check_registry = BranchPythonOperator(
        task_id="check_registry_status",
        python_callable=_check_registry_status,
    )

    # 2. Path A: Train from Scratch (Tabula Rasa)
    full_training = DockerOperator(
        task_id="run_full_training",
        image="dag-pytorch-model-training:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        environment={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "TRAINING_MODE": "SCRATCH"  # Directs the python script
        },
        command="python train_transformer.py",
        docker_url="unix://var/run/docker.sock",
        device_requests=[
            DeviceRequest(count=-1, capabilities=[['gpu']]) 
        ],
        pool="single_gpu",
        retries=0,
        execution_timeout=timedelta(hours=4), # Generous timeout for 30 epochs
    )

    # 3. Path B: Train Incrementally (Replay Buffer)
    incremental_training = DockerOperator(
        task_id="run_incremental_training",
        image="dag-pytorch-model-training:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        environment={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "TRAINING_MODE": "INCREMENTAL"  # Directs the python script
        },
        command="python train_transformer.py",
        docker_url="unix://var/run/docker.sock",
        device_requests=[
            DeviceRequest(count=-1, capabilities=[['gpu']]) 
        ],
        pool="single_gpu",
        retries=0,
        execution_timeout=timedelta(minutes=45), # Tight timeout for 5 epochs
    )

    # Define the DAG Graph
    check_registry >> [full_training, incremental_training]