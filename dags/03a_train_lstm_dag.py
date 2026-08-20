"""Airflow DAG: weekly Conv-LSTM training.

The word "airflow" above is load-bearing. DAG_DISCOVERY_SAFE_MODE only parses files
that contain both "airflow" and "dag", and this file's imports mention neither, so
without it the DAG silently disappears from the UI with no import error.
"""
from training_dag_factory import build_training_dag

dag = build_training_dag(
    dag_id="03a_train_pytorch_lstm",
    model_name="Weather_Forecaster_FastLSTM",
    script="train_lstm.py",
    description="Intelligent weekly training for the Conv-LSTM forecaster",
    tags=["lstm", "ml", "pytorch", "training", "gpu", "smart-routing"],
)
