"""Airflow DAG: weekly Transformer training.

The word "airflow" above is load-bearing. DAG_DISCOVERY_SAFE_MODE only parses files
that contain both "airflow" and "dag", and this file's imports mention neither, so
without it the DAG silently disappears from the UI with no import error.
"""
from training_dag_factory import build_training_dag

dag = build_training_dag(
    dag_id="03b_train_pytorch_transformer",
    model_name="Weather_Forecaster_Transformer",
    script="train_transformer.py",
    description="Intelligent weekly training for the Transformer forecaster",
    tags=["ml", "pytorch", "training", "gpu", "attention", "smart-routing"],
)
