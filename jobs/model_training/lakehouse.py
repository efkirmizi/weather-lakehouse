"""Shared Iceberg access and artifact conventions for the model_training jobs.

Kept free of torch imports so the light jobs (drift monitoring, batch inference)
do not pay for them.
"""
import os
from pyiceberg.catalog import load_catalog

CATALOG_URI = "http://nessie:19120/iceberg/main"
S3_ENDPOINT = "http://minio:9000"

# The registry entry carries the float "pytorch" flavor (warm-start + benchmark).
# trainer.py logs the serving graph beside it in the same run under this name,
# and batch_inference.py executes that graph through ONNX Runtime.
ONNX_ARTIFACT_NAME = "onnx_model"
ONNX_OPSET = 17


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
