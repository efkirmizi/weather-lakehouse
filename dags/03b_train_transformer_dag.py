import os
from datetime import datetime
from airflow import DAG
from airflow.datasets import Dataset
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import DeviceRequest

ml_features_dataset = Dataset("iceberg://nessie.weather.ml_features")

with DAG(
    dag_id="03b_train_pytorch_transformer",
    description="Trains Transformer Attention model on Iceberg via PyArrow",
    schedule=[ml_features_dataset],
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["ml", "pytorch", "training", "gpu", "attention"],
) as dag:

    train_transformer_task = DockerOperator(
        task_id="run_transformer_training",
        image="dag-pytorch-model-training:1.0",
        api_version="auto",
        auto_remove=True,
        network_mode="lakehouse-net",
        environment={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        },
        command="python train_transformer.py",
        docker_url="unix://var/run/docker.sock",
        device_requests=[
            DeviceRequest(count=-1, capabilities=[['gpu']]) 
        ],
        pool="single_gpu",
        retries=0,
    )