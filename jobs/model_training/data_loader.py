import datetime
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from lakehouse import load_iceberg_catalog, scan_ordered

logger = logging.getLogger("ML_Training")

# Fixed by default so the promotion benchmark is scored on the same holdout every
# run instead of a fresh random one.
DEFAULT_SEED = 1337

# Chronological three-way split: train | validation | test, oldest to newest.
# Validation drives early stopping, so it cannot also be the promotion benchmark -
# the challenger would be judged on data it was tuned against. The test slice is
# never read during training.
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Incremental runs mix recent windows with a replay buffer. Capped relative to the
# recent set: an uncapped 5% of all history buried the 14 days we actually want to
# adapt to under ~38k historical windows, i.e. under 1% of every batch.
REPLAY_PER_RECENT = 4
RECENT_WINDOWS = 24 * 14

# Training runs on a weekly cron, not on a feature-table event, so nothing stops it
# from retraining on data that stopped arriving days ago. The archive API lags a
# day or two and 01 runs daily, so anything past three days means 01 is broken.
DEFAULT_MAX_FEATURE_AGE_HOURS = int(os.getenv("MAX_FEATURE_AGE_HOURS", "72"))


class IcebergTimeSeriesDataset(Dataset):
    """Sliding windows over the feature table, skipping any window that spans a gap.

    Rows are sliced positionally, so a missing hour would silently produce a sequence
    that jumps in time. The ETL's data-quality gate only warns about gaps (the
    Open-Meteo archive genuinely has holes in the older decades), so the filtering
    has to happen here.
    """

    def __init__(self, table_name: str, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len
        window_len = seq_len + pred_len

        logger.info(f"Connecting to Iceberg catalog to load {table_name}...")
        catalog = load_iceberg_catalog()

        table = catalog.load_table(tuple(table_name.split(".")))
        arrow_table = scan_ordered(table, ("timestamp", "features"))

        features = np.stack(arrow_table.column("features").to_numpy(zero_copy_only=False))
        self.data = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))

        # Sorted ascending by scan_ordered, so the last row is the newest.
        self.latest_timestamp = arrow_table.column("timestamp")[-1].as_py()

        timestamps = arrow_table.column("timestamp").to_numpy(zero_copy_only=False).astype("datetime64[h]")
        self.valid_starts = self._contiguous_starts(timestamps, window_len)

        candidates = max(0, len(timestamps) - window_len + 1)
        dropped = candidates - len(self.valid_starts)
        if dropped:
            logger.warning(f"Skipping {dropped} window(s) that span a gap in the hourly series.")
        logger.info(f"Dataset ready. Generated {len(self.valid_starts)} contiguous chronological windows.")

    @staticmethod
    def _contiguous_starts(timestamps: np.ndarray, window_len: int) -> np.ndarray:
        """Start indices whose window covers exactly window_len consecutive hours."""
        if len(timestamps) < window_len:
            return np.empty(0, dtype=np.int64)
        span = timestamps[window_len - 1:] - timestamps[: len(timestamps) - window_len + 1]
        return np.flatnonzero(span == np.timedelta64(window_len - 1, "h")).astype(np.int64)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = int(self.valid_starts[idx])
        end_idx = start_idx + self.seq_len
        target_end = end_idx + self.pred_len

        x = self.data[start_idx:end_idx]
        y = self.data[end_idx:target_end]
        return x, y


def assert_features_fresh(latest_timestamp, max_age_hours: int) -> None:
    """Refuses to train on a feature table that has stopped being updated.

    Deliberately loud: a silent retrain on week-old data produces a model that looks
    fine by every metric and is quietly blind to everything since the ETL broke.
    """
    if latest_timestamp.tzinfo is None:
        latest_timestamp = latest_timestamp.replace(tzinfo=datetime.timezone.utc)

    age = datetime.datetime.now(datetime.timezone.utc) - latest_timestamp
    if age > datetime.timedelta(hours=max_age_hours):
        raise RuntimeError(
            f"weather.ml_features is stale: newest row is {latest_timestamp} "
            f"({age} old, limit {max_age_hours}h). Check 01_extract_weather_data. "
            f"Raise MAX_FEATURE_AGE_HOURS to override deliberately."
        )
    logger.info(f"Feature freshness OK: newest row {latest_timestamp} ({age} old).")


def chronological_split(indices, seq_len, pred_len,
                        val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION):
    """Cuts ordered window indices into train / validation / test at two points in time.

    Consecutive windows share seq_len-1 of their timesteps, so a random split hands
    the later sets near-duplicates of the training rows and reports an optimistic
    loss. Cutting chronologically and discarding the windows that straddle each cut
    keeps the three sets genuinely disjoint.
    """
    ordered = sorted(indices)
    total = len(ordered)
    n_test = int(total * test_fraction)
    n_val = int(total * val_fraction)

    if n_test == 0 or n_val == 0:
        raise ValueError(
            f"Not enough windows ({total}) to carve out validation and test splits."
        )

    test_start = total - n_test
    val_start = test_start - n_val

    # A window starting at i spans [i, i + seq_len + pred_len), so the last few
    # windows of each block reach into the next one. Drop them.
    gap = seq_len + pred_len
    train_indices = ordered[: max(0, val_start - gap)]
    val_indices = ordered[val_start: max(val_start, test_start - gap)]
    test_indices = ordered[test_start:]

    if not train_indices or not val_indices:
        raise ValueError(
            f"Split gaps of {gap} windows consumed a whole block "
            f"({total} windows available). Need more history."
        )

    return train_indices, val_indices, test_indices


def _select_indices(total_windows, is_incremental, seed):
    if not is_incremental:
        logger.info(f"Scratch mode: Training on ALL {total_windows} historical windows.")
        return list(range(total_windows))

    recent_indices = list(range(max(0, total_windows - RECENT_WINDOWS), total_windows))
    historical_pool = list(range(0, max(0, total_windows - RECENT_WINDOWS)))

    budget = len(recent_indices) * REPLAY_PER_RECENT
    sample_size = min(int(len(historical_pool) * 0.05), budget)
    # Seeded so a rerun of the same DAG run trains on the same replay buffer.
    rng = random.Random(seed)
    replay_indices = rng.sample(historical_pool, sample_size) if historical_pool else []

    logger.info(
        f"Incremental mode: {len(recent_indices)} recent + {sample_size} replay windows "
        f"(capped at {REPLAY_PER_RECENT}x recent)."
    )
    return sorted(recent_indices + replay_indices)


def get_dataloaders(table_name, seq_len, pred_len, batch_size, is_incremental,
                    seed=DEFAULT_SEED, max_age_hours=None):
    """Returns (train_loader, val_loader, test_loader).

    Training uses train + val; the promotion benchmark uses test and nothing else.
    Pass max_age_hours to refuse stale data - training does, evaluation does not,
    because a promotion decision on an already-registered model is still valid.
    """
    full_dataset = IcebergTimeSeriesDataset(table_name, seq_len=seq_len, pred_len=pred_len)

    if max_age_hours is not None:
        assert_features_fresh(full_dataset.latest_timestamp, max_age_hours)

    selected_indices = _select_indices(len(full_dataset), is_incremental, seed)
    train_indices, val_indices, test_indices = chronological_split(selected_indices, seq_len, pred_len)
    logger.info(
        f"Chronological split: {len(train_indices)} train / {len(val_indices)} validation "
        f"/ {len(test_indices)} test windows."
    )

    def loader(indices, shuffle):
        return DataLoader(
            Subset(full_dataset, indices), batch_size=batch_size, shuffle=shuffle,
            num_workers=4, pin_memory=True, persistent_workers=True,
        )

    return loader(train_indices, True), loader(val_indices, False), loader(test_indices, False)
