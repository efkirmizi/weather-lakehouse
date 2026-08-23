"""Watermark handling in the ETL. Runs inside dag-pyspark-etl (PYTHONPATH=/opt/spark/work-dir)."""
import pandas as pd
import pytest

from weather_etl import (
    DEFAULT_PARAMS,
    EARLIEST_DATA_DATE,
    HOURLY_VARIABLES,
    drop_already_ingested,
    generate_date_chunks,
    run_data_quality_checks,
)


def _day(date_str, hours=24, start_hour=0):
    return pd.DataFrame({
        "timestamp": pd.date_range(f"{date_str} {start_hour:02d}:00", periods=hours,
                                   freq="h", tz="UTC"),
        "temperature_c": [1.0] * hours,
    })


def test_nothing_is_dropped_before_the_table_exists():
    df = _day("2026-08-20")
    assert len(drop_already_ingested(df, None)) == 24


def test_hours_already_in_the_table_are_dropped():
    """The API returns whole days. When transform_weather_data drops a NaN tail the
    watermark stops short of 23:00, so the next run re-fetches that whole day - and the
    load is a bare append with no key, so those hours become duplicate rows."""
    df = _day("2026-08-20")
    watermark = pd.Timestamp("2026-08-20 18:00", tz="UTC")

    kept = drop_already_ingested(df, watermark)

    assert len(kept) == 5
    assert kept["timestamp"].min() == pd.Timestamp("2026-08-20 19:00", tz="UTC")


def test_a_fully_covered_chunk_is_emptied_not_re_appended():
    df = _day("2026-08-20")
    assert drop_already_ingested(df, pd.Timestamp("2026-08-20 23:00", tz="UTC")).empty


def test_chunks_start_from_the_archive_epoch_when_there_is_no_watermark():
    chunks = generate_date_chunks(None)
    assert chunks and chunks[0][0] == EARLIEST_DATA_DATE


def test_chunks_resume_on_the_day_after_a_complete_watermark():
    watermark = pd.Timestamp("2026-08-18 23:00", tz="UTC")
    chunks = generate_date_chunks(watermark)
    assert chunks and chunks[0][0] == "2026-08-19"


def test_a_mid_day_watermark_refetches_its_own_day():
    """Deliberate: the API only serves whole days, so the re-fetch is unavoidable.
    drop_already_ingested is what stops it becoming duplicate rows."""
    watermark = pd.Timestamp("2026-08-18 18:00", tz="UTC")
    chunks = generate_date_chunks(watermark)
    assert chunks and chunks[0][0] == "2026-08-18"


def _valid_frame(**overrides):
    """One physically plausible hour, with every column the ETL now ingests."""
    row = {
        "timestamp": pd.Timestamp("2026-08-20 12:00", tz="UTC"),
        "temperature_c": 25.0,
        "humidity_percent": 55.0,
        "precipitation_mm": 0.0,
        "wind_speed_kmh": 12.0,
        "pressure_msl_hpa": 1013.0,
        "dew_point_c": 15.0,
        "cloud_cover_percent": 40.0,
        "shortwave_radiation_wm2": 700.0,
        "soil_temperature_c": 27.0,
        "soil_moisture_m3m3": 0.21,
        "wind_direction_deg": 210.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_the_request_list_is_generated_from_the_variable_table():
    """The SDK returns variables positionally, so the request order and the column
    order must come from one declaration or they drift apart silently."""
    assert DEFAULT_PARAMS["hourly"] == [api for api, _, _ in HOURLY_VARIABLES]


def test_a_valid_hour_passes_every_gate():
    run_data_quality_checks(_valid_frame())


@pytest.mark.parametrize("column,bad", [
    ("pressure_msl_hpa", 1200.0),
    ("cloud_cover_percent", 140.0),
    ("shortwave_radiation_wm2", -5.0),
    ("soil_moisture_m3m3", 1.4),
    ("wind_direction_deg", 400.0),
])
def test_each_new_column_has_a_physical_bound(column, bad):
    with pytest.raises(ValueError, match=column):
        run_data_quality_checks(_valid_frame(**{column: bad}))


def test_dew_point_above_temperature_is_rejected():
    """Air cannot hold more moisture than saturation. A violation means the two
    columns have been misaligned - which no per-column bound check can see, because
    both values are individually plausible."""
    with pytest.raises(ValueError, match="dew point"):
        run_data_quality_checks(_valid_frame(temperature_c=10.0, dew_point_c=18.0))


def test_saturation_noise_is_tolerated():
    """At saturation ERA5 puts the dew point a hair above the temperature."""
    run_data_quality_checks(_valid_frame(temperature_c=10.0, dew_point_c=10.3))
