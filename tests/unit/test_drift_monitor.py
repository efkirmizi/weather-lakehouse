"""Anchor selection for the drift baseline. Runs inside dag-pytorch-model-training."""
import datetime

import pandas as pd

import drift_monitor


class _Arrow:
    def __init__(self, timestamps):
        self._ts = list(timestamps)
        self.num_rows = len(self._ts)

    def column(self, _name):
        series = pd.Series(self._ts)
        return type("Col", (), {"to_pandas": lambda _self: series})()


class FakeTable:
    """Records every scan so a test can assert which of them were bounded."""

    def __init__(self, timestamps, visible_to_filtered_scan=True):
        self._ts = timestamps
        self._visible = visible_to_filtered_scan
        self.filters = []

    def scan(self, row_filter=None, selected_fields=None):
        self.filters.append(row_filter)
        rows = self._ts if (row_filter is None or self._visible) else []
        return type("S", (), {"to_arrow": lambda _self: _Arrow(rows)})()


def _hours(n):
    return list(pd.date_range("2026-08-01", periods=n, freq="h", tz="UTC"))


def test_the_anchor_probe_is_bounded():
    """ml_features grows forever. Reading its whole timestamp column just to find the
    newest row defeats the point of the bounded climatology scan it exists to set up."""
    table = FakeTable(_hours(48))

    drift_monitor.latest_timestamp(table)

    assert table.filters[0] is not None


def test_the_anchor_is_the_newest_row():
    ts = _hours(48)
    assert drift_monitor.latest_timestamp(FakeTable(ts)) == pd.Timestamp(ts[-1])


def test_a_stale_table_falls_back_to_an_unbounded_scan():
    """If nothing landed inside the probe window, 01 has stopped - still report the
    real anchor rather than pretending the table is empty."""
    table = FakeTable(_hours(48), visible_to_filtered_scan=False)

    result = drift_monitor.latest_timestamp(table)

    assert len(table.filters) == 2 and table.filters[1] is None
    assert result == pd.Timestamp(_hours(48)[-1])


def test_an_empty_table_has_no_anchor():
    assert drift_monitor.latest_timestamp(FakeTable([])) is None
