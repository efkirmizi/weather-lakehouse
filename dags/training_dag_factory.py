"""Shared shape for the model training DAGs.

03a and 03b were byte-identical apart from the model name and the script they run,
which meant every fix had to be applied twice - and more than once only landed in one
of them. This module holds the single copy; the DAG files just supply the differences.
"""
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from docker.types import DeviceRequest

from alerting import DEFAULT_ARGS
from credentials import s3_env

TRAINING_IMAGE = "dag-pytorch-model-training:1.0"
MLFLOW_URL = "http://mlflow:5000"

# PyTorch DataLoader workers communicate over /dev/shm; Docker's 64MB default is
# not enough for num_workers=4 and shows up as opaque worker crashes.
SHM_SIZE = 2 * 1024 ** 3


def _make_branch_callable(model_name: str):
    def _check_registry_status(**context):
        """Pings MLflow to see if the model exists, and routes accordingly.

        A caller can force a from-scratch run with conf {"TRAINING_MODE": "SCRATCH"}.
        drift_monitor.py does exactly that: when the input distribution has shifted,
        warm-starting from weights fitted on the pre-drift data is the thing we are
        trying to get away from.
        """
        dag_run = context.get("dag_run")
        forced_mode = (dag_run.conf or {}).get("TRAINING_MODE") if dag_run else None
        if forced_mode == "SCRATCH":
            print("Caller requested TRAINING_MODE=SCRATCH. Taking full training path.")
            return "run_full_training"

        url = f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get?name={model_name}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Model {model_name} found in registry! Taking incremental path.")
                return "run_incremental_training"
            print(f"Model {model_name} not found (Status {response.status_code}). Taking full training path.")
            return "run_full_training"
        except Exception as e:
            print(f"Failed to reach MLflow ({e}). Defaulting to full training.")
            return "run_full_training"

    return _check_registry_status


def _training_task(task_id: str, script: str, mode: str, timeout: timedelta) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=TRAINING_IMAGE,
        api_version="auto",
        auto_remove="success",
        network_mode="lakehouse-net",
        environment={
            **s3_env(),
            "TRAINING_MODE": mode,
            # Lets a single DAG run smoke-test the pipeline quickly without
            # touching the production epoch defaults.
            "TRAINING_EPOCHS": "{{ dag_run.conf.get('TRAINING_EPOCHS', '') }}",
        },
        command=f"python {script}",
        docker_url="unix://var/run/docker.sock",
        device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
        pool="single_gpu",
        shm_size=SHM_SIZE,
        retries=0,
        execution_timeout=timeout,
    )


def build_training_dag(dag_id: str, model_name: str, script: str, description: str, tags: list) -> DAG:
    with DAG(
        dag_id=dag_id,
        description=description,
        schedule="0 2 * * 0",
        start_date=datetime(2026, 8, 14),
        catchup=False,
        tags=tags,
        default_args=DEFAULT_ARGS,
        is_paused_upon_creation=False,
    ) as dag:

        check_registry = BranchPythonOperator(
            task_id="check_registry_status",
            python_callable=_make_branch_callable(model_name),
        )

        full_training = _training_task(
            "run_full_training", script, "SCRATCH", timedelta(hours=4),
        )
        incremental_training = _training_task(
            "run_incremental_training", script, "INCREMENTAL", timedelta(minutes=45),
        )

        evaluate_and_promote_task = DockerOperator(
            task_id="evaluate_and_promote_model",
            image=TRAINING_IMAGE,
            api_version="auto",
            auto_remove="success",
            network_mode="lakehouse-net",
            environment={
                **s3_env(),
                "MODEL_REGISTRY_NAME": model_name,
            },
            command="python evaluate_and_promote.py",
            docker_url="unix://var/run/docker.sock",
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
            pool="single_gpu",
            shm_size=SHM_SIZE,
            retries=0,
            # Runs whichever training branch was taken; the other is skipped, not failed.
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            execution_timeout=timedelta(minutes=15),
        )

        check_registry >> [full_training, incremental_training] >> evaluate_and_promote_task

    return dag
