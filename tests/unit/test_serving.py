"""Serving-layer logic. Runs inside the serving image."""
import datetime

import pandas as pd
import pytest

import api
from api import _utc_iso, denormalize


def _fake_tables(monkeypatch, **frames):
    """Point get_iceberg_table at in-memory frames, keyed by table name."""
    class FakeTable:
        def __init__(self, df): self._df = df
        def scan(self, **kw):
            df = self._df
            arrow = type("A", (), {"to_pandas": lambda _self: df.copy()})()
            return type("S", (), {"to_arrow": lambda _self: arrow})()

    monkeypatch.setattr(api, "get_iceberg_table", lambda name: FakeTable(frames[name]))
    monkeypatch.setattr(api, "refresh_scaling_parameters", lambda force=False: None)
    api._scaling_cache["temperature"] = (0.0, 1.0)


@pytest.fixture
def hours():
    return pd.date_range("2026-08-20", periods=24, freq="h", tz="UTC")


def test_naive_input_gains_a_utc_offset():
    """Every table stores timestamptz; pyiceberg rejects a zone-less literal."""
    out = _utc_iso(datetime.datetime(2026, 8, 19, 12, 0, 0))
    assert out.endswith("+00:00")


def test_aware_input_is_converted_not_relabelled():
    plus_three = datetime.timezone(datetime.timedelta(hours=3))
    out = _utc_iso(datetime.datetime(2026, 8, 19, 12, 0, 0, tzinfo=plus_three))
    assert out.startswith("2026-08-19T09:00:00")
    assert out.endswith("+00:00")


def test_pandas_timestamp_is_accepted():
    assert _utc_iso(pd.Timestamp("2026-08-19 12:00:00")).endswith("+00:00")


def test_denormalize_inverts_the_standardization():
    from api import _scaling_cache
    _scaling_cache["temperature"] = (14.0, 8.0)
    assert denormalize(0.0) == 14.0
    assert denormalize(1.0) == 22.0


def test_denormalize_falls_back_to_identity_for_unknown_features():
    assert denormalize(3.5, feature="not_a_feature") == 3.5


def _forecast_rows(hours, model, created_at, value=0.5, version="1"):
    return [{"forecast_timestamp": h, "predicted_features": [value, 0.0, 0.0, 0.0],
             "model_name": model, "model_version": version, "created_at": created_at}
            for h in hours]


def test_latest_forecast_keeps_every_model_not_just_the_newest_batch(monkeypatch, hours):
    """The idempotency guard skips a champion whose forecast for this window already
    exists, so the two models' rows carry different created_at values. Filtering on one
    global max(created_at) drops whichever model was written earlier - which is the
    entire point of a multi-model comparison chart."""
    df = pd.DataFrame(
        _forecast_rows(hours, "Weather_Forecaster_FastLSTM", pd.Timestamp("2026-08-20 01:00", tz="UTC"))
        + _forecast_rows(hours, "Weather_Forecaster_Transformer", pd.Timestamp("2026-08-20 02:00", tz="UTC"))
    )
    _fake_tables(monkeypatch, forecast_predictions=df)

    charted = {k for row in api.get_latest_forecast() for k in row if k != "forecast_timestamp"}
    assert charted == {"Weather_Forecaster_FastLSTM", "Weather_Forecaster_Transformer"}


def test_latest_forecast_uses_the_newest_batch_per_model(monkeypatch, hours):
    """A superseded forecast from the same model must not win over its newer one."""
    df = pd.DataFrame(
        _forecast_rows(hours, "M", pd.Timestamp("2026-08-20 01:00", tz="UTC"), value=1.0, version="1")
        + _forecast_rows(hours, "M", pd.Timestamp("2026-08-20 05:00", tz="UTC"), value=2.0, version="2")
    )
    _fake_tables(monkeypatch, forecast_predictions=df)

    assert {row["M"] for row in api.get_latest_forecast()} == {2.0}


def test_residuals_dedup_keeps_the_newest_write_whatever_the_scan_order(monkeypatch, hours):
    """Iceberg guarantees no row order, so `keep='last'` on a raw scan is arbitrary.
    Here the stale row sorts last, exactly as a compaction could leave it."""
    fresh = _forecast_rows(hours, "M", pd.Timestamp("2026-08-20 05:00", tz="UTC"), value=1.0)
    stale = _forecast_rows(hours, "M", pd.Timestamp("2026-08-20 01:00", tz="UTC"), value=99.0)
    preds = pd.DataFrame(fresh + stale)          # stale physically last
    actuals = pd.DataFrame([{"timestamp": h, "features": [1.0, 0.0, 0.0, 0.0]} for h in hours])
    _fake_tables(monkeypatch, forecast_predictions=preds, ml_features=actuals)

    metrics = api.get_residuals()
    assert len(metrics) == 1
    assert metrics[0]["mae"] == 0.0          # the fresh row predicted the actual exactly


def test_a_failed_scaling_refresh_backs_off_instead_of_retrying_every_request(monkeypatch):
    """Before the scaling table exists, the TTL check never short-circuited: every
    single request re-hit the catalog. The failure branch already recorded a timestamp
    that nothing read."""
    calls = []

    def boom(name):
        calls.append(name)
        raise RuntimeError("table does not exist yet")

    monkeypatch.setattr(api, "get_iceberg_table", boom)
    monkeypatch.setattr(api, "_scaling_loaded_at", 0.0)
    monkeypatch.setattr(api, "_scaling_is_real", False)

    api.refresh_scaling_parameters()
    api.refresh_scaling_parameters()
    api.refresh_scaling_parameters()

    assert len(calls) == 1
