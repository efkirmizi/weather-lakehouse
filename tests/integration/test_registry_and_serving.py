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
                       PRED_LEN, S3_ENDPOINT, SEQ_LEN, scan_ordered)

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


def test_the_served_forecast_continues_the_observed_series(catalog):
    """The next 24 hours have to look like weather that follows the last observed
    hour, not like numbers from somewhere else.

    An earlier version of this test asserted only that some served value sat outside
    +/-5, on the reasoning that normalized units cluster near zero. That passes every
    August in this location and would raise a false alarm some January, when the real
    temperature is near zero too - a test whose verdict depends on the season is not
    a test. Anchoring to the last observation instead is what makes it year-round.

    A 20 degree band is far wider than a real 24-hour swing here and still far
    narrower than the gap between a z-score and a temperature whenever the ambient
    value is not close to the training mean. It also catches a model that has simply
    come apart, whatever the cause.
    """
    observations = catalog.load_table(("weather", "observations"))
    column = scan_ordered(observations, ("timestamp", "temperature_c")).column("temperature_c")
    last_observed = column.to_pylist()[-1]

    payload = requests.get(f"{API}/api/v1/forecast/latest", timeout=60).json()
    served = [row[m] for row in payload for m in REGISTERED_MODELS if m in row]

    assert served, "no forecast values to check"
    stray = [v for v in served if abs(v - last_observed) >= 20.0]
    assert not stray, (
        f"last observed {last_observed:.1f} degC, but the forecast contains "
        f"{stray[:5]} - either de-normalization was skipped or the model is broken"
    )
