"""Scoring the promotion benchmark. Runs inside dag-pytorch-model-training
(PYTHONPATH=/app)."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluate_and_promote import (
    can_score,
    champion_alone_cannot_score,
    horizon_rmse_celsius,
    resolve_temperature_std,
    score_predictors,
)
from lakehouse import OUTPUT_CHANNELS

PRED = 24


class CountingLoader:
    """A DataLoader stand-in that records how often it was iterated.

    The test block is ~113k windows, so how many times it is walked is not an
    implementation detail - it is the cost of the whole evaluation task.
    """

    def __init__(self, batches):
        self._batches = batches
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _loader(n_batches=3, batch=4, seq=72, feats=16, seed=0):
    """Batches whose target is the last PRED hours of their own context, narrowed to
    the forecast channels."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch, seq, feats, generator=generator)
        batches.append((x, x[:, -PRED:, :OUTPUT_CHANNELS].clone()))
    return CountingLoader(batches)


def _perfect(x):
    return x[:, -PRED:, :OUTPUT_CHANNELS]


def _off_by_one(x):
    return x[:, -PRED:, :OUTPUT_CHANNELS] + 1.0


def test_each_predictor_is_scored_independently():
    """One shared loop must not let one predictor's errors leak into another's.

    Scored alone these are trivial; scored together they are the whole risk of
    walking the loader once - a single accumulator, or a window_mse list appended
    to by every predictor, produces numbers that look plausible and are wrong.
    """
    scores = score_predictors(
        {"perfect": _perfect, "off_by_one": _off_by_one}, _loader(), nn.SmoothL1Loss()
    )

    assert scores["perfect"].rmse == 0.0
    assert scores["perfect"].mae == 0.0
    assert scores["perfect"].loss == 0.0

    assert scores["off_by_one"].rmse == 1.0
    assert scores["off_by_one"].mae == 1.0


def test_the_loader_is_walked_once_for_every_predictor():
    """The reason this function exists.

    Champion, challenger and two naive baselines used to be four separate passes
    over the same 113k windows, inside a task bounded at 15 minutes.
    """
    loader = _loader()

    score_predictors(
        {"a": _perfect, "b": _off_by_one, "c": _perfect, "d": _off_by_one},
        loader, nn.SmoothL1Loss(),
    )

    assert loader.passes == 1


def test_window_errors_are_one_per_window_and_stay_aligned():
    """promotion_verdict pairs the two arrays positionally, so order and length
    have to match across predictors or the paired test compares unrelated hours."""
    scores = score_predictors(
        {"perfect": _perfect, "off_by_one": _off_by_one}, _loader(), nn.SmoothL1Loss()
    )

    assert scores["perfect"].window_mse.shape == (12,)   # 3 batches x 4 windows
    assert scores["off_by_one"].window_mse.shape == (12,)
    assert (scores["perfect"].window_mse == 0.0).all()
    assert (scores["off_by_one"].window_mse == 1.0).all()


def test_an_empty_predictor_set_still_consumes_nothing():
    """Defensive: NAIVE_FORECASTS is configurable, and an empty mapping must not
    walk 113k windows to produce nothing."""
    loader = _loader()

    assert score_predictors({}, loader, nn.SmoothL1Loss()) == {}
    assert loader.passes == 0


def _horizon_loader(n_batches=2, batch=4, seq=72):
    """Targets whose error grows with the horizon, by construction."""
    batches = []
    for _ in range(n_batches):
        x = torch.zeros(batch, seq, 16)
        y = torch.zeros(batch, PRED, OUTPUT_CHANNELS)
        batches.append((x, y))
    return CountingLoader(batches)


def _ramp(x):
    """Predicts hour k as k+1, so the error at horizon k is exactly k+1."""
    steps = torch.arange(1, PRED + 1, dtype=torch.float32)
    return steps.view(1, PRED, 1).expand(x.shape[0], PRED, OUTPUT_CHANNELS)


def test_per_horizon_rmse_is_reported_for_every_forecast_hour():
    scores = score_predictors({"ramp": _ramp}, _horizon_loader(), nn.SmoothL1Loss())
    horizon = scores["ramp"].horizon_rmse

    assert horizon.shape == (PRED,)
    # error at horizon k is k+1 by construction
    np.testing.assert_allclose(horizon, np.arange(1, PRED + 1), rtol=1e-5)


def test_per_horizon_rmse_converts_to_celsius_by_the_published_std():
    """The gate works in normalized units; the number a person reads has to be a
    temperature. scaling_parameters is what makes that conversion exact."""
    scores = score_predictors({"ramp": _ramp}, _horizon_loader(), nn.SmoothL1Loss())
    celsius = horizon_rmse_celsius(scores["ramp"], temperature_std=8.164)
    np.testing.assert_allclose(celsius, np.arange(1, PRED + 1) * 8.164, rtol=1e-5)


def test_a_flat_forecast_has_a_flat_horizon_curve():
    """Guards against accumulating across horizons instead of per horizon."""
    scores = score_predictors(
        {"off_by_one": lambda x: torch.ones(x.shape[0], PRED, OUTPUT_CHANNELS)},
        _horizon_loader(), nn.SmoothL1Loss(),
    )
    np.testing.assert_allclose(scores["off_by_one"].horizon_rmse, np.ones(PRED), rtol=1e-5)


class _ScalingArrow:
    """Enough of pyarrow's Table surface for load_scaling_parameters: one column
    per field, each read out with to_pylist()."""

    def __init__(self, rows):
        self._rows = rows

    def column(self, name):
        values = self._rows[name]
        return type("Col", (), {"to_pylist": lambda _self: values})()


class _ScalingTable:
    def __init__(self, rows):
        self._rows = rows

    def scan(self):
        return type("S", (), {"to_arrow": lambda _self: _ScalingArrow(self._rows)})()


class _FakeCatalog:
    """A catalog stand-in whose load_table() either returns a canned
    scaling_parameters table or raises, reaching both resolve_temperature_std
    branches without a live Iceberg catalog."""

    def __init__(self, table=None, error=None):
        self._table = table
        self._error = error

    def load_table(self, _name):
        if self._error is not None:
            raise self._error
        return self._table


def test_resolve_temperature_std_reads_the_std_not_the_mean():
    """An index slip here (mean instead of std) would make every horizon figure
    wrong by a plausible-looking factor and nothing would complain - the two values
    are kept far apart so a slip like that fails loudly."""
    catalog = _FakeCatalog(table=_ScalingTable({
        "feature_name": ["temperature", "humidity"],
        "mean_value": [11.0, 61.0],
        "std_value": [8.164, 20.5],
    }))

    std, unit = resolve_temperature_std(lambda: catalog)

    assert std == 8.164
    assert unit == "degC"


def test_resolve_temperature_std_falls_back_when_scaling_parameters_is_unreadable():
    """A benchmark run must not fail just because a diagnostic table is unreachable."""
    catalog = _FakeCatalog(error=RuntimeError("scaling_parameters not found"))

    std, unit = resolve_temperature_std(lambda: catalog)

    assert std == 1.0
    assert unit == "normalized"


def test_resolve_temperature_std_falls_back_when_the_catalog_itself_is_unreachable():
    """Regression test: resolve_temperature_std(load_iceberg_catalog()) used to
    evaluate the catalog acquisition before entering the function, outside any
    guard, so a transient REST-catalog or credentials failure propagated straight
    out of evaluate_and_promote() instead of falling back. Taking a loader and
    calling it inside the try is what keeps acquisition - not just the read -
    inside the fallback."""
    def load_catalog():
        raise RuntimeError("could not connect to the REST catalog")

    std, unit = resolve_temperature_std(load_catalog)

    assert std == 1.0
    assert unit == "normalized"


def test_a_model_that_matches_the_current_vector_can_be_scored():
    assert can_score(lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS), _loader())


def test_a_model_from_an_older_feature_schema_is_detected_not_raised():
    """Loading such a champion succeeds; the first forward pass is where the width
    mismatch surfaces. Letting it escape aborts the whole promotion task, which is
    how a schema change turns into a red DAG instead of a decision."""
    def four_channel_model(x):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x16 and 4x32)")

    assert can_score(four_channel_model, _loader()) is False


def test_the_check_consumes_only_one_batch():
    """113k windows are not needed to learn that a model does not fit."""
    loader = _loader()
    can_score(lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS), loader)
    assert loader.passes == 1


def test_only_the_champion_failing_licenses_promoting_around_it():
    """The challenger was just trained against the current vector, so it is
    compatible by construction - its success on this exact batch is what tells a
    schema casualty apart from a transient failure that would sink either model."""
    def broken(x):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x16 and 4x32)")
    healthy = lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS)

    assert champion_alone_cannot_score(broken, healthy, _loader()) is True


def test_a_healthy_champion_is_never_routed_around():
    healthy = lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS)

    assert champion_alone_cannot_score(healthy, healthy, _loader()) is False


def test_when_both_models_fail_the_task_fails_loudly_instead_of_promoting():
    """This is the case that used to auto-promote: a CUDA OOM or a flaky kernel
    sinks both models identically to a schema casualty, and would have swapped
    out a working champion under an alias asserting it cannot run, with no
    comparison ever performed. The exception is raised before either alias is
    set, so a caller that lets it propagate sets none."""
    def broken(x):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="Neither"):
        champion_alone_cannot_score(broken, broken, _loader())
