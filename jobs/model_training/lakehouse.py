"""Shared Iceberg access and artifact conventions for the model_training jobs.

Kept free of torch imports so the light jobs (drift monitoring, batch inference)
do not pay for them.
"""
import os
from pyiceberg.catalog import load_catalog

CATALOG_URI = "http://nessie:19120/iceberg/main"
S3_ENDPOINT = "http://minio:9000"
# Was a literal in trainer.py, batch_inference.py and evaluate_and_promote.py.
MLFLOW_TRACKING_URI = "http://mlflow:5000"

# The registry entry carries the float "pytorch" flavor (warm-start + benchmark).
# trainer.py logs the serving graph beside it in the same run under this name,
# and batch_inference.py executes that graph through ONNX Runtime.
ONNX_ARTIFACT_NAME = "onnx_model"
ONNX_OPSET = 17

# The window and channel geometry every job in this image has to agree on. These were
# literals in three separate files; a mismatch does not raise anything, it just slices
# a different window and reports a confident wrong number.
SEQ_LEN = 72
PRED_LEN = 24

# ml_features carries 16 channels (see 02's channel declaration and the published
# scaling_parameters); only the first four - temperature, humidity, precipitation,
# wind_speed - are forecast. The rest are exogenous or deterministic inputs.
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 4
# Position of temperature, which is what the serving layer de-normalizes and what the
# per-horizon benchmark reports.
TEMPERATURE_CHANNEL = 0


def load_iceberg_catalog():
    return load_catalog(
        "default",
        **{
            "type": "rest",
            "uri": CATALOG_URI,
            "s3.endpoint": S3_ENDPOINT,
            "s3.access-key-id": os.environ["AWS_ACCESS_KEY_ID"],
            "s3.secret-access-key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "s3.path-style-access": "true"
        }
    )


def scan_ordered(table, fields):
    """Scans an Iceberg table and returns the rows in chronological order.

    Iceberg makes no ordering guarantee across data files, and rewrite_data_files
    (05_lakehouse_maintenance) can reshuffle them at any time. Every consumer here
    slices sequences positionally, so this sort is what makes those slices correct.
    """
    arrow_table = table.scan(selected_fields=fields).to_arrow()
    return arrow_table.sort_by([("timestamp", "ascending")])


def load_scaling_parameters(catalog):
    """{feature_name: (mean, std)} exactly as 02 published them.

    The gate scores in normalized units, which are unreadable. This is the same table
    the serving API inverts with, so a number reported here and a number on the
    dashboard mean the same thing.
    """
    table = catalog.load_table(("weather", "scaling_parameters"))
    arrow = table.scan().to_arrow()
    return {
        name: (float(mean), float(std))
        for name, mean, std in zip(
            arrow.column("feature_name").to_pylist(),
            arrow.column("mean_value").to_pylist(),
            arrow.column("std_value").to_pylist(),
        )
    }
