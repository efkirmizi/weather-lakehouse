"""The registry and the API that reads from it.

The promotion gate moves an alias; batch inference resolves that alias to a version
and runs its ONNX graph; the API serves what batch inference wrote. Each step is unit
tested against a stub of the next one. Nothing until now checked that the chain holds
together in the running system - that the alias points at a version that exists, that
the version carries the artifact the serving path needs, and that a forecast actually
comes out of the other end.
"""
import os

import mlflow
import pytest
import requests

from lakehouse import (INPUT_CHANNELS, MLFLOW_TRACKING_URI, ONNX_ARTIFACT_NAME,
                       PRED_LEN, S3_ENDPOINT, SEQ_LEN)

API = "http://lakehouse-serving:8000"
REGISTERED_MODELS = ["Weather_Forecaster_FastLSTM", "Weather_Forecaster_Transformer"]


@pytest.fixture(scope="session")
def client():
    # list_artifacts talks to the object store directly rather than through the
    # tracking server, so without the endpoint override boto3 resolves s3://mlflow
    # against real AWS - where the answer is "that access key does not exist",
    # which reads like a credentials problem and is not one. Every job in this
    # image sets the same variable for the same reason.
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.tracking.MlflowClient()


@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_every_model_has_a_champion_that_resolves(client, model_name):
    """An alias in MLflow can outlive the version it names. batch_inference resolves
    '@champion' on every run and would fail there instead of here."""
    champion = client.get_model_version_by_alias(name=model_name, alias="champion")

    assert champion.version
    assert client.get_model_version(model_name, champion.version).status == "READY"


@pytest.mark.parametrize("model_name", REGISTERED_MODELS)
def test_the_champion_graph_loads_with_the_geometry_the_contract_declares(client, model_name):
    """Loaded through the exact URI batch_inference uses, then checked against the
    constants in lakehouse.py.

    Worth doing the expensive way rather than listing artifacts: under MLflow 3 the
    graph is a logged model, not a run artifact, so a listing shows only 'quantized'
    and would fail while the serving path works perfectly. Loading is also the only
    way to see the geometry - the graph is a compiled artifact sitting in object
    storage, and if SEQ_LEN or INPUT_CHANNELS ever drift from what this champion was
    trained on, nothing else notices. batch_inference would feed it a 72x16 window,
    ONNX Runtime would reject the shape, and the forecast DAG would fail at 03:00.
    """
    champion = client.get_model_version_by_alias(name=model_name, alias="champion")
    graph = mlflow.onnx.load_model(f"runs:/{champion.run_id}/{ONNX_ARTIFACT_NAME}")

    inputs = graph.graph.input
    assert len(inputs) == 1, f"expected a single input tensor, got {len(inputs)}"

    dims = [d.dim_param or d.dim_value for d in inputs[0].type.tensor_type.shape.dim]
    assert dims[1:] == [SEQ_LEN, INPUT_CHANNELS], (
        f"{model_name}@champion (v{champion.version}) takes {dims[1:]}, but this image's "
        f"contract says every window is {[SEQ_LEN, INPUT_CHANNELS]}"
    )


def test_the_api_is_healthy():
    response = requests.get(f"{API}/health", timeout=30)

    assert response.status_code == 200


def test_the_api_serves_a_full_horizon_from_every_champion():
    """The shape the dashboard draws. A forecast short of PRED_LEN hours means batch
    inference wrote a partial window, which no unit test can see because each one
    stubs the table the others read."""
    payload = requests.get(f"{API}/api/v1/forecast/latest", timeout=60).json()

    assert len(payload) == PRED_LEN, f"expected {PRED_LEN} forecast hours, got {len(payload)}"
    for model_name in REGISTERED_MODELS:
        assert all(model_name in row for row in payload), f"{model_name} missing from the forecast"


def test_the_served_temperatures_are_in_degrees_not_in_normalized_units():
    """The one failure this whole scaling_parameters contract exists to prevent, seen
    from the outside: if the API's inverse transform is skipped or reads the wrong
    row, the numbers stay plausible-looking floats near zero instead of raising."""
    payload = requests.get(f"{API}/api/v1/forecast/latest", timeout=60).json()

    values = [row[m] for row in payload for m in REGISTERED_MODELS if m in row]
    assert values, "no forecast values to check"
    assert any(abs(v) > 5.0 for v in values), (
        f"every forecast sits within +/-5 of zero ({min(values):.2f} to {max(values):.2f}), "
        "which is what normalized units look like when de-normalization is skipped"
    )
