"""Windowing and splitting logic. Runs inside dag-pytorch-model-training (PYTHONPATH=/app)."""
import numpy as np
import pytest

from data_loader import (
    REPLAY_PER_RECENT,
    RECENT_WINDOWS,
    IcebergTimeSeriesDataset,
    _select_indices,
    chronological_split,
)

SEQ, PRED = 72, 24
GAP = SEQ + PRED


def test_split_blocks_are_chronological_and_disjoint():
    train, val, test = chronological_split(range(1000), SEQ, PRED)

    assert train and val and test
    # ordered oldest -> newest
    assert max(train) < min(val) < max(val) < min(test)
    # a window starting at i occupies [i, i+GAP); no block may reach into the next
    assert max(train) + GAP <= min(val)
    assert max(val) + GAP <= min(test)


def test_split_test_block_is_untouched_by_training():
    train, val, test = chronological_split(range(10000), SEQ, PRED)
    assert not (set(train) | set(val)) & set(test)


def test_split_rejects_too_little_history():
    with pytest.raises(ValueError):
        chronological_split(range(5), SEQ, PRED)


def test_split_survives_sparse_non_contiguous_indices():
    """Incremental runs feed in a sparse mix of recent and replay indices."""
    sparse = sorted(list(range(0, 100000, 37))[:2000])
    train, val, test = chronological_split(sparse, SEQ, PRED)
    assert max(train) + GAP <= min(val)
    assert max(val) + GAP <= min(test)


def _hours(*offsets):
    base = np.datetime64("2026-01-01T00", "h")
    return np.array([base + np.timedelta64(o, "h") for o in offsets])


def test_contiguous_starts_accepts_an_unbroken_run():
    starts = IcebergTimeSeriesDataset._contiguous_starts(_hours(*range(10)), 4)
    assert starts.tolist() == [0, 1, 2, 3, 4, 5, 6]


def test_contiguous_starts_rejects_windows_spanning_a_gap():
    # hour 4 is missing
    starts = IcebergTimeSeriesDataset._contiguous_starts(_hours(0, 1, 2, 3, 5, 6, 7, 8), 4)
    assert starts.tolist() == [0, 4]


def test_contiguous_starts_handles_too_little_data():
    assert IcebergTimeSeriesDataset._contiguous_starts(_hours(0, 1), 4).size == 0


def test_replay_buffer_is_capped_relative_to_recent_windows():
    selected = _select_indices(100_000, is_incremental=True, seed=1337)
    replay = len(selected) - RECENT_WINDOWS
    assert replay <= RECENT_WINDOWS * REPLAY_PER_RECENT
    # the whole point: recent data must not be a rounding error in the batch mix
    assert RECENT_WINDOWS / len(selected) > 0.15


def test_replay_selection_is_deterministic():
    assert _select_indices(100_000, True, 1337) == _select_indices(100_000, True, 1337)
    assert _select_indices(100_000, True, 1337) != _select_indices(100_000, True, 7)


def test_scratch_mode_uses_every_window():
    assert _select_indices(500, is_incremental=False, seed=1337) == list(range(500))


def test_fresh_features_pass_the_guard():
    from data_loader import assert_features_fresh
    import datetime as dt
    recent = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=25)
    assert_features_fresh(recent, max_age_hours=72)  # must not raise


def test_stale_features_are_refused():
    from data_loader import assert_features_fresh
    import datetime as dt
    ancient = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=8)
    with pytest.raises(RuntimeError, match="stale"):
        assert_features_fresh(ancient, max_age_hours=72)


def test_naive_timestamps_are_treated_as_utc():
    from data_loader import assert_features_fresh
    import datetime as dt
    naive_recent = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=1)
    assert_features_fresh(naive_recent, max_age_hours=72)
