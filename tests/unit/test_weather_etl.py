"""Watermark handling in the ETL. Runs inside dag-pyspark-etl (PYTHONPATH=/opt/spark/work-dir)."""
import pandas as pd
import pytest

from weather_etl import EARLIEST_DATA_DATE, drop_already_ingested, generate_date_chunks


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
