"""Timestamp literals for pyiceberg row filters. Runs inside the serving image."""
import datetime

import pandas as pd

from api import _utc_iso, denormalize


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
