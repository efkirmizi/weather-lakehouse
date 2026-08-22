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


import math

from feature_engineering import (
    CYCLICAL_CHANNELS,
    FEATURE_COLS,
    SCALED_COLUMNS,
    SERVING_FEATURE_NAMES,
    cyclical_pair,
    scaling_parameter_rows,
)


def test_the_first_four_channels_never_move():
    """The serving API de-normalizes channel 0 as temperature from an image that
    cannot import this layout at all, and drift_monitor/evaluate_and_promote read it
    through TEMPERATURE_CHANNEL, which only agrees with this file by convention.
    Moving these silently re-labels every forecast the dashboard draws."""
    assert SERVING_FEATURE_NAMES[:4] == [
        "temperature", "humidity", "precipitation", "wind_speed"
    ]


def test_the_vector_is_sixteen_channels_and_names_are_unique():
    assert len(SERVING_FEATURE_NAMES) == 16
    assert len(set(SERVING_FEATURE_NAMES)) == 16


def test_scaled_columns_and_feature_cols_stay_in_step():
    """FEATURE_COLS drives standardization; SERVING_FEATURE_NAMES drives the published
    table. They are two views of one list and must not drift."""
    assert FEATURE_COLS == [column for column, _ in SCALED_COLUMNS]
    assert SERVING_FEATURE_NAMES[:len(SCALED_COLUMNS)] == [n for _, n in SCALED_COLUMNS]


def test_cyclical_channels_come_last_as_sin_cos_pairs():
    tail = SERVING_FEATURE_NAMES[len(SCALED_COLUMNS):]
    expected = [f"{name}_{part}" for name, _, _ in CYCLICAL_CHANNELS
                for part in ("sin", "cos")]
    assert tail == expected


def _period(prefix):
    """The period CYCLICAL_CHANNELS actually declares for a channel, read from the
    source rather than retyped, so a typo'd period there fails this test instead of
    the test quietly checking its own, disconnected number."""
    return next(period for name, period, _ in CYCLICAL_CHANNELS if name == prefix)


def test_a_compass_bearing_wraps_without_a_seam():
    """359 degrees and 1 degree are two degrees apart. On a raw scale they are the
    two extremes, which is why direction is encoded as sin/cos at all."""
    period = _period("wind_dir")
    near_north = cyclical_pair(359.0, period)
    just_past_north = cyclical_pair(1.0, period)
    distance = math.dist(near_north, just_past_north)
    assert distance < 0.05


def test_new_years_eve_and_new_years_day_are_neighbours():
    period = _period("doy")
    end = cyclical_pair(365.0, period)
    start = cyclical_pair(1.0, period)
    assert math.dist(end, start) < 0.05


def test_midnight_and_twenty_three_hundred_are_neighbours():
    period = _period("hour")
    assert math.dist(cyclical_pair(23.0, period), cyclical_pair(0.0, period)) < 0.3


def test_cyclical_values_stay_inside_the_unit_circle():
    """They are published with identity scaling, so they have to already be bounded."""
    for value in (0.0, 90.0, 180.0, 270.0, 359.0):
        for component in cyclical_pair(value, 360.0):
            assert -1.0 <= component <= 1.0


def _stats_with_distinct_values():
    """One distinct (mean, std) per standardized column, so a column transposed
    with its neighbour would show up as the wrong pair at the wrong index instead
    of accidentally matching."""
    return {column: (float(i), float(i) + 1.0) for i, (column, _) in enumerate(SCALED_COLUMNS)}


def test_scaling_parameter_rows_cover_all_sixteen_channels():
    assert len(scaling_parameter_rows(_stats_with_distinct_values())) == 16


def test_the_cyclical_tail_is_published_with_identity_scaling():
    """Channels 10-15 are already bounded in [-1, 1] and need no inverse transform,
    but a consumer of this table must not have to know which channels it silently
    omits - so they still get a row, with mean=0, std=1."""
    rows = scaling_parameter_rows(_stats_with_distinct_values())
    for _, _, mean, std in rows[10:16]:
        assert (mean, std) == (0.0, 1.0)


def test_feature_index_is_contiguous_and_aligned_with_serving_names():
    """write_scaling_parameters would still emit 16 rows with all the right names
    even if the index computation were wrong - row count and the name column give
    no warning. This is the assertion that actually catches a skipped or
    duplicated index, which is what load_published_stats and the serving
    dashboard both trust to line up with SERVING_FEATURE_NAMES."""
    rows = scaling_parameter_rows(_stats_with_distinct_values())
    assert [index for _, index, _, _ in rows] == list(range(16))
    assert [name for name, _, _, _ in rows] == SERVING_FEATURE_NAMES


import datetime

import pytest
from pyspark.sql import SparkSession

from feature_engineering import standardize


@pytest.fixture(scope="module")
def spark():
    """A real local Spark session - the only thing in this file that actually
    executes the Spark expression at feature_engineering.py:171-174 rather than
    its cyclical_pair twin. cyclical_pair is never called by production code, so
    every assertion above this point would keep passing even if the Spark
    expression itself used the wrong datetime part or the wrong period.

    JVM startup makes this fixture, and the test that uses it, a few seconds
    slower than the rest of this file. That is the cost of actually running the
    code being tested rather than its Python stand-in, and is not something a
    later cleanup pass should "optimise" away.
    """
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("test_feature_engineering_standardize")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


def _context_row(month, day, hour, wind_deg):
    """A source row for standardize(). The timestamp is chosen so day-of-month !=
    day-of-year (true of every month but January) - the property that would catch
    F.dayofyear silently becoming F.dayofmonth - and each row gets a distinct hour
    and wind bearing so a period swapped between two cyclical channels (hour's
    24.0 and doy's 365.25, say) also produces a mismatch rather than a coincidence."""
    row = {c: 0.0 for c in FEATURE_COLS}
    row["timestamp"] = datetime.datetime(2026, month, day, hour, tzinfo=datetime.timezone.utc)
    row["wind_direction_deg"] = wind_deg
    return row


CONTEXT_ROWS = [
    _context_row(3, 10, 6, 45.0),
    _context_row(6, 21, 18, 190.0),
    _context_row(11, 5, 23, 300.0),
    _context_row(8, 22, 15, 359.0),
]


def test_cyclical_channels_match_the_python_oracle(spark):
    """The one test in the suite that runs the production Spark code path for
    channels 10-15 instead of asserting on cyclical_pair alone. Compares
    standardize()'s output against cyclical_pair fed the same wind bearing, hour
    and day-of-year - an F.dayofyear-for-F.dayofmonth slip or a period swapped
    between two channels changes the Spark side only, so either would show up
    here as a mismatch, without which it would ship in all 759,000 rows unnoticed."""
    stats = {c: (0.0, 1.0) for c in FEATURE_COLS}
    df = spark.createDataFrame(CONTEXT_ROWS)

    # Ordered rather than looked up by timestamp: a Spark TimestampType round-trips
    # through collect() as a naive datetime, which would never equal the
    # timezone-aware Python datetimes above, so a dict keyed on it would just miss
    # every row. Sorting both sides by the same key and pairing positionally sidesteps
    # that entirely.
    expected_rows = sorted(CONTEXT_ROWS, key=lambda r: r["timestamp"])
    actual_rows = standardize(df, stats).orderBy("timestamp").collect()

    for expected, actual in zip(expected_rows, actual_rows):
        features = actual["features"]
        timestamp = expected["timestamp"]
        expected_wind = cyclical_pair(expected["wind_direction_deg"], 360.0)
        expected_hour = cyclical_pair(timestamp.hour, 24.0)
        expected_doy = cyclical_pair(timestamp.timetuple().tm_yday, 365.25)

        assert features[10:12] == pytest.approx(expected_wind, abs=1e-5)
        assert features[12:14] == pytest.approx(expected_hour, abs=1e-5)
        assert features[14:16] == pytest.approx(expected_doy, abs=1e-5)
