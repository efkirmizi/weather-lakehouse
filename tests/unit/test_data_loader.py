"""Windowing and splitting logic. Runs inside dag-pytorch-model-training (PYTHONPATH=/app)."""
import numpy as np
import pytest

from data_loader import (
    REPLAY_PER_RECENT,
    RECENT_WINDOWS,
    IcebergTimeSeriesDataset,
    chronological_split,
    plan_windows,
    split_windows,
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


def test_replay_selection_is_deterministic():
    """A rerun of the same DAG run must train on the same replay buffer."""
    same = plan_windows(100_000, SEQ, PRED, True, seed=1337)[0]
    assert same == plan_windows(100_000, SEQ, PRED, True, seed=1337)[0]
    assert same != plan_windows(100_000, SEQ, PRED, True, seed=7)[0]


def test_scratch_mode_trains_on_the_whole_train_block():
    train_block, val_block, test_block, adapt = split_windows(100_000, SEQ, PRED)
    train, val, test = plan_windows(100_000, SEQ, PRED, is_incremental=False)
    assert (train, val, test) == (train_block, val_block, test_block)
    assert not set(train) & set(adapt)


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


# --- the composed split -------------------------------------------------------
#
# The blind spot that let a real bug through: every test above checks _select_indices
# or chronological_split on its own, and both were individually correct. The defect
# lived in their composition - the recent windows sort last, so the split pushed all
# 336 of them into the val and test slices that training discards, and an incremental
# run fitted nothing but its historical replay buffer.

REAL_TABLE_WINDOWS = 759_330  # weather.ml_features as of 2026-08-20


def test_incremental_run_trains_on_the_newest_windows():
    """The 14 days an incremental run exists to adapt to must reach the optimizer."""
    train, _, _ = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=True)
    newest = set(range(REAL_TABLE_WINDOWS - RECENT_WINDOWS, REAL_TABLE_WINDOWS))
    assert newest <= set(train)


def test_recent_windows_are_not_a_rounding_error_in_the_incremental_mix():
    """Why REPLAY_PER_RECENT exists, asserted on the mix training actually receives."""
    train, _, _ = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=True)
    recent = sum(1 for i in train if i >= REAL_TABLE_WINDOWS - RECENT_WINDOWS)
    assert recent / len(train) > 0.15


def test_incremental_training_never_touches_the_promotion_test_block():
    """The gate scores on the test block, so no mode may fit any window in it."""
    *_, gate_block = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=False)
    train, val, _ = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=True)
    assert not (set(train) | set(val)) & set(gate_block)


def test_the_benchmark_block_is_identical_in_both_modes():
    """evaluate_and_promote always asks for the scratch split; a mode-dependent test
    block would score champion and challenger on different windows."""
    *_, scratch_block = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=False)
    *_, incremental_block = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=True)
    assert scratch_block == incremental_block


def test_replay_buffer_stays_within_its_cap():
    """Uncapped, 5% of 86 years of history drowns the 14 days being adapted to."""
    train, _, _ = plan_windows(REAL_TABLE_WINDOWS, SEQ, PRED, is_incremental=True)
    replay = sum(1 for i in train if i < REAL_TABLE_WINDOWS - RECENT_WINDOWS)
    assert replay <= RECENT_WINDOWS * REPLAY_PER_RECENT


def test_adapt_block_is_the_newest_windows_and_disjoint_from_the_rest():
    train, val, test, adapt = split_windows(REAL_TABLE_WINDOWS, SEQ, PRED)
    assert adapt == list(range(REAL_TABLE_WINDOWS - RECENT_WINDOWS, REAL_TABLE_WINDOWS))
    assert max(train) < min(val) and max(val) < min(test) and max(test) < min(adapt)


import torch

from lakehouse import OUTPUT_CHANNELS


def test_the_target_carries_only_the_forecast_channels():
    """x keeps every input channel; y keeps only what the model predicts. Returning
    all 16 as the target would have the model fitting the calendar it was handed."""
    dataset = IcebergTimeSeriesDataset.__new__(IcebergTimeSeriesDataset)
    dataset.seq_len, dataset.pred_len = 3, 2
    dataset.valid_starts = np.array([0], dtype=np.int64)
    dataset.data = torch.arange(16 * 5, dtype=torch.float32).reshape(5, 16)

    x, y = dataset[0]

    assert x.shape == (3, 16)
    assert y.shape == (2, OUTPUT_CHANNELS)
    # y must be the first columns of the rows that follow x, not a reshaped slice
    assert torch.equal(y, dataset.data[3:5, :OUTPUT_CHANNELS])
