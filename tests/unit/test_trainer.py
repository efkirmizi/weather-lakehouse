"""Epoch budget and warm-start selection. Runs inside dag-pytorch-model-training
(PYTHONPATH=/app)."""
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import trainer
from trainer import resolve_epochs


@pytest.fixture(autouse=True)
def clear_override():
    os.environ.pop("TRAINING_EPOCHS", None)
    yield
    os.environ.pop("TRAINING_EPOCHS", None)


def test_defaults_when_unset():
    assert resolve_epochs(is_incremental=False, scratch=10, incremental=2) == 10
    assert resolve_epochs(is_incremental=True, scratch=10, incremental=2) == 2


def test_override_is_honoured():
    os.environ["TRAINING_EPOCHS"] = "1"
    assert resolve_epochs(False, scratch=10, incremental=2) == 1


def test_override_tolerates_whitespace():
    os.environ["TRAINING_EPOCHS"] = "  3 "
    assert resolve_epochs(False, scratch=10, incremental=2) == 3


@pytest.mark.parametrize("bad", ["abc", "0", "-5", ""])
def test_bad_override_falls_back_to_the_default(bad):
    """A typo in a DAG conf must never silently weaken the production budget."""
    os.environ["TRAINING_EPOCHS"] = bad
    assert resolve_epochs(False, scratch=10, incremental=2) == 10


def test_warm_start_resumes_from_the_champion_not_the_newest_version(monkeypatch):
    """A challenger that failed the gate still registers as the newest version.

    Warm-starting from `max(version)` resumes from a model the pipeline has already
    rejected, and compounds it every week. The live registry showed exactly this:
    Weather_Forecaster_FastLSTM sat at champion v1 with a rejected v5 on top.
    """
    requested = []

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert alias == "champion"
            return SimpleNamespace(version="1")

        def search_model_versions(self, _filter):
            return [SimpleNamespace(version=str(v)) for v in range(1, 6)]

    def fake_load_model(uri, map_location=None):
        requested.append(uri)
        return SimpleNamespace(state_dict=lambda: {"w": "champion-weights"})

    monkeypatch.setattr(trainer, "MlflowClient", FakeClient)
    monkeypatch.setattr(trainer.mlflow.pytorch, "load_model", fake_load_model)

    weights = trainer.get_champion_weights("Weather_Forecaster_FastLSTM", device="cpu")

    assert requested == ["models:/Weather_Forecaster_FastLSTM/1"]
    assert weights == {"w": "champion-weights"}


def test_warm_start_falls_back_to_scratch_when_no_champion_exists(monkeypatch):
    """First ever run: nothing to resume from, and that is not an error."""
    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")

    monkeypatch.setattr(trainer, "MlflowClient", FakeClient)
    assert trainer.get_champion_weights("Weather_Forecaster_FastLSTM", device="cpu") is None


def test_warm_start_talks_to_the_tracking_server_not_the_local_store(monkeypatch):
    """MlflowClient() built before set_tracking_uri() resolves to the container's own
    ./mlruns, which is empty in an ephemeral job container.

    Not hypothetical: the incremental branch fired twice in this pipeline and both runs
    logged "No existing model found in MLflow Registry" while the branch operator -
    which queries the REST API directly - had just found the model. The artifact
    download needs the MinIO endpoint for the same reason.
    """
    monkeypatch.delenv("MLFLOW_S3_ENDPOINT_URL", raising=False)
    seen = {}
    monkeypatch.setattr(trainer.mlflow, "set_tracking_uri",
                        lambda uri: seen.__setitem__("uri", uri))

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            seen["uri_at_lookup"] = seen.get("uri")
            seen["s3_at_lookup"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
            raise RuntimeError("no champion registered")

    monkeypatch.setattr(trainer, "MlflowClient", FakeClient)
    trainer.get_champion_weights("Weather_Forecaster_FastLSTM", device="cpu")

    assert seen["uri_at_lookup"] == "http://mlflow:5000"
    assert seen["s3_at_lookup"] == "http://minio:9000"


def _tiny(width):
    """A stand-in for a real forecaster: all warm_start cares about is whether the
    tensor shapes line up."""
    return nn.Sequential(nn.Linear(width, 4))


def test_warm_start_loads_weights_that_fit():
    source = _tiny(8)
    target = _tiny(8)

    assert trainer.warm_start(target, source.state_dict()) is True
    assert torch.equal(target[0].weight, source[0].weight)


def test_warm_start_declines_weights_from_a_different_architecture():
    """Change a hyperparameter - a wider d_model, another layer, a different input
    width - and the champion's tensors no longer fit the model just built.
    load_state_dict raises, and the weekly INCREMENTAL run dies as a DAG failure
    instead of doing the obviously right thing: the architecture changed, so there is
    nothing to resume from, so train from scratch. get_champion_weights already
    applies that fallback when the champion cannot be fetched at all; this is the
    same fallback one step later."""
    champion = _tiny(8).state_dict()
    model = _tiny(16)
    untouched = model[0].weight.clone()

    assert trainer.warm_start(model, champion) is False
    assert torch.equal(model[0].weight, untouched), "a declined warm start must leave the fresh init alone"


def test_warm_start_declines_when_there_is_no_champion():
    assert trainer.warm_start(_tiny(8), None) is False
