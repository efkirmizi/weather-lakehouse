"""Rebuild-vs-append decision. Runs inside dag-pyspark-feature-engineering."""
from feature_engineering import FEATURE_COLS, stats_drifted

TOL = 0.01


def _stats(**overrides):
    base = {c: (10.0, 5.0) for c in FEATURE_COLS}
    base.update(overrides)
    return base


def test_identical_statistics_do_not_rebuild():
    assert stats_drifted(_stats(), _stats(), TOL) is False


def test_small_move_stays_incremental():
    moved = _stats(**{FEATURE_COLS[0]: (10.025, 5.0)})  # 0.5% of a std
    assert stats_drifted(_stats(), moved, TOL) is False


def test_mean_move_beyond_tolerance_rebuilds():
    moved = _stats(**{FEATURE_COLS[0]: (10.10, 5.0)})   # 2% of a std
    assert stats_drifted(_stats(), moved, TOL) is True


def test_std_move_beyond_tolerance_rebuilds():
    moved = _stats(**{FEATURE_COLS[1]: (10.0, 5.10)})
    assert stats_drifted(_stats(), moved, TOL) is True


def test_near_zero_mean_does_not_rebuild_on_a_large_relative_change():
    """Precipitation's mean is ~0.07: measuring drift against the mean would rebuild
    on every drizzle, which is why the yardstick is the standard deviation."""
    precip = FEATURE_COLS[2]
    published = _stats(**{precip: (0.05, 0.30)})
    current = _stats(**{precip: (0.052, 0.30)})   # +4% of the mean, 0.67% of a std
    assert stats_drifted(published, current, TOL) is False


def test_near_zero_mean_still_rebuilds_on_a_real_shift():
    precip = FEATURE_COLS[2]
    published = _stats(**{precip: (0.05, 0.30)})
    current = _stats(**{precip: (0.10, 0.30)})    # 17% of a std
    assert stats_drifted(published, current, TOL) is True
