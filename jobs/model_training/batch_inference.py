import os
import sys
import logging
import datetime
import pyarrow as pa
import numpy as np
import onnxruntime as ort

from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual
from pyiceberg.schema import Schema
from pyiceberg.types import TimestamptzType, ListType, FloatType, StringType, NestedField
from pyiceberg.partitioning import PartitionSpec, PartitionField
from pyiceberg.transforms import DayTransform

import mlflow.onnx
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from lakehouse import load_iceberg_catalog, scan_ordered, ONNX_ARTIFACT_NAME

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Dynamic_Batch_Inference")

# Configuration
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "Weather_Forecaster_")
SEQ_LEN = 72
PRED_LEN = 24

def discover_champions(client: MlflowClient, prefix: str):
    """Dynamically searches MLflow for any model with the given prefix holding a '@champion' alias."""
    active_champions = []
    registered_models = client.search_registered_models()

    for rm in registered_models:
        if rm.name.startswith(prefix):
            aliases = rm.aliases or {}
            if "champion" in aliases:
                champ_version = aliases["champion"]
                active_champions.append((rm.name, champ_version))
                logger.info(f"Discovered Active Champion: '{rm.name}' at Version v{champ_version}")
            else:
                logger.warning(f"Model '{rm.name}' exists but has no '@champion' alias. Skipping.")

    return active_champions

def _providers():
    """Requests the CUDA provider only when this build actually registered it.

    requirements.txt pins the CPU-only onnxruntime on purpose (see the note there),
    so this normally resolves to CPU. Asking for a provider the build does not carry
    would fail at session construction instead of degrading.
    """
    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if "CUDAExecutionProvider" not in available:
        logger.warning("CUDAExecutionProvider unavailable; running inference on CPU.")
    return preferred or ["CPUExecutionProvider"]


def load_predictor(client: MlflowClient, model_name: str, version: str):
    """Returns a callable that maps a float32 batch to predictions.

    The registry entry carries the 'pytorch' flavor (needed for warm-starting and for
    the champion/challenger benchmark). trainer.py logs the serving graph beside it in
    the same run as an ONNX artifact, which is what we prefer to execute.

    Versions registered before that artifact existed - or by anything other than
    trainer.py - fall back to the registered PyTorch model, so a serving run is never
    lost to a missing side artifact.
    """
    run_id = client.get_model_version(model_name, version).run_id

    # Only a missing artifact is an expected fallback. A broken ONNX Runtime, an OOM
    # or a network failure must not be laundered into "no ONNX artifact" - that would
    # silently pin serving to the slow path forever. Those propagate to main(), which
    # reports the model as failed.
    try:
        onnx_model = mlflow.onnx.load_model(f"runs:/{run_id}/{ONNX_ARTIFACT_NAME}")
    except (MlflowException, OSError) as e:
        logger.warning(
            f"No ONNX artifact for '{model_name}' v{version} ({e}). "
            "Falling back to the registered PyTorch model; retrain to get the ONNX path."
        )
        # `from ... import ... as` on purpose: a bare `import mlflow.pytorch` here would
        # make `mlflow` a function-local name and turn the mlflow.onnx call above into
        # an UnboundLocalError, silently disabling the ONNX path entirely.
        import torch
        from mlflow import pytorch as mlflow_pytorch

        model = mlflow_pytorch.load_model(f"models:/{model_name}/{version}", map_location="cpu")
        model.eval()

        def predict(batch):
            with torch.no_grad():
                return model(torch.from_numpy(batch)).cpu().numpy()

        return predict

    session = ort.InferenceSession(onnx_model.SerializeToString(), providers=_providers())
    input_name = session.get_inputs()[0].name
    logger.info(f"Using ONNX serving graph for '{model_name}' v{version}.")
    return lambda batch: session.run(None, {input_name: batch})[0]

PREDICTIONS_TABLE = ("weather", "forecast_predictions")


def build_context_window(catalog):
    """The last SEQ_LEN feature rows, plus the timestamps they imply for the forecast.

    Rejects a discontinuous window: the rows are sliced positionally, so a missing
    hour would hand the model a sequence that silently jumps in time. A wrong
    forecast is worse than a missing one.
    """
    feature_table = catalog.load_table(("weather", "ml_features"))
    arrow_table = scan_ordered(feature_table, ("timestamp", "features"))

    features_col = arrow_table.column("features").to_pylist()
    timestamps_col = arrow_table.column("timestamp").to_pylist()

    if len(features_col) < SEQ_LEN:
        logger.error(f"Insufficient historical data! Expected at least {SEQ_LEN} records, found {len(features_col)}.")
        sys.exit(1)

    context_timestamps = timestamps_col[-SEQ_LEN:]
    expected_span = datetime.timedelta(hours=SEQ_LEN - 1)
    actual_span = context_timestamps[-1] - context_timestamps[0]
    if actual_span != expected_span:
        logger.error(
            f"Context window is not contiguous: {SEQ_LEN} rows span {actual_span} "
            f"instead of {expected_span}. Refusing to forecast from a sequence with gaps."
        )
        sys.exit(1)

    # Everything in the warehouse is timestamptz, so stay UTC-aware end to end:
    # pyiceberg rejects a filter literal whose zone-awareness differs from the column.
    last_known = context_timestamps[-1]
    if last_known.tzinfo is None:
        last_known = last_known.replace(tzinfo=datetime.timezone.utc)

    future_timestamps = [last_known + datetime.timedelta(hours=i) for i in range(1, PRED_LEN + 1)]
    x_input = np.array([features_col[-SEQ_LEN:]], dtype=np.float32)
    return x_input, last_known, future_timestamps


def open_predictions_table(catalog):
    try:
        return catalog.load_table(PREDICTIONS_TABLE)
    except Exception:
        logger.info("Table 'weather.forecast_predictions' does not exist. Creating schema & partitions...")
        schema = Schema(
            NestedField(1, "forecast_timestamp", TimestamptzType(), required=True),
            NestedField(2, "predicted_features", ListType(element_id=3, element_type=FloatType()), required=True),
            NestedField(4, "model_name", StringType(), required=True),
            NestedField(5, "model_version", StringType(), required=True),
            NestedField(6, "created_at", TimestamptzType(), required=True)
        )
        spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=DayTransform(), name="forecast_day"))
        return catalog.create_table("weather.forecast_predictions", schema=schema, partition_spec=spec)


def already_forecast(predictions_table, future_timestamps):
    """(model_name, model_version) pairs that already cover this exact forecast window.

    This DAG is dataset-triggered, and 02 emits its outlet even on a no-op run, so the
    same 24 hours get re-predicted from the same context on every trigger. Appending
    those again inflates forecast_predictions without bound and skews
    /api/v1/metrics/residuals, which averages over every stored copy.
    """
    window_filter = And(
        GreaterThanOrEqual("forecast_timestamp", future_timestamps[0].isoformat()),
        LessThanOrEqual("forecast_timestamp", future_timestamps[-1].isoformat()),
    )
    existing = predictions_table.scan(
        row_filter=window_filter,
        selected_fields=("model_name", "model_version"),
    ).to_arrow()

    return set(zip(existing.column("model_name").to_pylist(),
                   existing.column("model_version").to_pylist()))


def main():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()

    # 1. Dynamically Discover All Active Champions
    champions = discover_champions(client, MODEL_PREFIX)
    if not champions:
        logger.error(f"No active @champion models found matching prefix '{MODEL_PREFIX}'. Exiting inference.")
        sys.exit(0)

    logger.info(f"Found {len(champions)} active champion model(s).")

    # 2. Build the shared context window once
    logger.info("Connecting to Iceberg catalog to fetch context window...")
    catalog = load_iceberg_catalog()
    x_input, last_known, future_timestamps = build_context_window(catalog)
    logger.info(f"Context window ends at {last_known}; forecasting {future_timestamps[0]} .. {future_timestamps[-1]}.")

    # 3. Skip champions whose forecast for this window is already stored
    predictions_table = open_predictions_table(catalog)
    covered = already_forecast(predictions_table, future_timestamps)
    pending = [(name, str(version)) for name, version in champions if (name, str(version)) not in covered]

    for name, version in champions:
        if (name, str(version)) in covered:
            logger.info(f"'{name}' v{version} already has a forecast for this window. Skipping.")

    if not pending:
        logger.info("Every champion is already up to date for this context window. Nothing to write.")
        return

    # 4. Run Inference Across the Remaining Champions
    all_forecast_timestamps = []
    all_predicted_features = []
    all_model_names = []
    all_model_versions = []
    all_inference_times = []

    now_utc = datetime.datetime.now(datetime.timezone.utc)

    for model_name, version in pending:
        logger.info(f"Executing forward pass for '{model_name}@champion' (v{version})...")
        try:
            predict = load_predictor(client, model_name, version)
            predictions = predict(x_input)

            # Extract the raw 24-hour sequence (predictions shape: [1, 24, 4])
            pred_list = predictions[0].tolist()

            all_forecast_timestamps.extend(future_timestamps)
            all_predicted_features.extend(pred_list)
            all_model_names.extend([model_name] * PRED_LEN)
            all_model_versions.extend([str(version)] * PRED_LEN)
            all_inference_times.extend([now_utc] * PRED_LEN)

            logger.info(f"Successfully generated predictions for '{model_name}'.")

        except Exception as e:
            logger.error(f"Failed to generate predictions for '{model_name}': {e}", exc_info=True)
            continue

    if not all_model_names:
        logger.error("All models failed during inference. Nothing to write.")
        sys.exit(1)

    pa_schema = pa.schema([
        pa.field("forecast_timestamp", pa.timestamp('us', tz='UTC'), nullable=False),
        pa.field("predicted_features", pa.list_(pa.field("element", pa.float32(), nullable=False)), nullable=False),
        pa.field("model_name", pa.string(), nullable=False),
        pa.field("model_version", pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp('us', tz='UTC'), nullable=False)
    ])

    pa_table = pa.Table.from_arrays(
        [
            pa.array(all_forecast_timestamps, type=pa.timestamp('us', tz='UTC')),
            pa.array(all_predicted_features, type=pa.list_(pa.float32())),
            pa.array(all_model_names, type=pa.string()),
            pa.array(all_model_versions, type=pa.string()),
            pa.array(all_inference_times, type=pa.timestamp('us', tz='UTC'))
        ],
        schema=pa_schema
    )

    logger.info("Writing batch forecasts to Iceberg table 'weather.forecast_predictions'...")
    predictions_table.append(pa_table)
    logger.info(
        f"Batch inference complete. Inserted {len(all_model_names)} forecast records "
        f"for {len(pending)} model(s); {len(champions) - len(pending)} were already current."
    )

if __name__ == "__main__":
    main()
