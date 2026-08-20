import os
import math
import time
import logging
import datetime
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import GreaterThanOrEqual, LessThanOrEqual, And

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Serving_API")

# The scaling table is written by 02_precompute_ml_features. Re-read it periodically
# instead of caching it forever at boot: the API usually starts before the pipeline
# has ever produced the table, and a permanent 0/1 fallback silently turns every
# temperature on the dashboard into a normalized number.
SCALING_TTL_SECONDS = 300
_scaling_cache = {"temperature": (0.0, 1.0)}
_scaling_loaded_at = 0.0
_scaling_is_real = False

# How far back to look. Both tables grow forever, so every read is bounded: without
# this, /metrics/residuals pulled all ~760k ml_features rows into pandas per request.
FORECAST_LOOKBACK_DAYS = int(os.getenv("FORECAST_LOOKBACK_DAYS", "7"))
RESIDUAL_LOOKBACK_DAYS = int(os.getenv("RESIDUAL_LOOKBACK_DAYS", "30"))


@lru_cache(maxsize=1)
def _catalog():
    """One catalog handle for the process; building it per request re-opens sessions."""
    return load_catalog(
        "default",
        **{
            "type": "rest",
            "uri": "http://nessie:19120/iceberg/main",
            "s3.endpoint": "http://minio:9000",
            "s3.access-key-id": os.environ["AWS_ACCESS_KEY_ID"],
            "s3.secret-access-key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "s3.path-style-access": "true"
        }
    )


def get_iceberg_table(table_name: str):
    return _catalog().load_table(("weather", table_name))


def _utc_iso(value) -> str:
    """UTC-aware ISO literal. Every table in the warehouse stores `timestamptz`, and
    pyiceberg rejects a literal whose zone-awareness differs from the column."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def _between(column: str, start, end):
    return And(
        GreaterThanOrEqual(column, _utc_iso(start)),
        LessThanOrEqual(column, _utc_iso(end)),
    )


def refresh_scaling_parameters(force: bool = False) -> None:
    """Reloads the normalization statistics if the cache is cold or stale."""
    global _scaling_loaded_at, _scaling_is_real

    fresh_enough = (time.time() - _scaling_loaded_at) < SCALING_TTL_SECONDS
    if not force and _scaling_is_real and fresh_enough:
        return

    try:
        table = get_iceberg_table("scaling_parameters")
        df = table.scan().to_arrow().to_pandas()
        _scaling_cache.update(
            {row["feature_name"]: (float(row["mean_value"]), float(row["std_value"]))
             for _, row in df.iterrows()}
        )
        _scaling_loaded_at = time.time()
        _scaling_is_real = True
        logger.info(f"Loaded scaling parameters for: {sorted(_scaling_cache)}")
    except Exception as e:
        _scaling_loaded_at = time.time()
        logger.warning(f"Could not load scaling parameters from Iceberg. Using 0/1 until it appears. ({e})")


def denormalize(value: float, feature: str = "temperature") -> float:
    mean_val, std_val = _scaling_cache.get(feature, (0.0, 1.0))
    return (value * std_val) + mean_val


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh_scaling_parameters(force=True)
    yield


app = FastAPI(title="Lakehouse Serving API", lifespan=lifespan)


# NOTE: every handler below is a plain `def`, not `async def`, on purpose. They do
# blocking S3 and pandas work; as coroutines they would run on the event loop and
# stall every other request - including /health, which is the container healthcheck.
# FastAPI runs sync handlers in a threadpool instead.

@app.get("/health")
def health():
    """Liveness probe. Also reports whether de-normalization is real or a fallback."""
    refresh_scaling_parameters()
    return {
        "status": "ok",
        "scaling_parameters_loaded": _scaling_is_real,
        "features": sorted(_scaling_cache),
    }


@app.get("/api/v1/forecast/latest")
def get_latest_forecast():
    try:
        refresh_scaling_parameters()
        now = datetime.datetime.now(datetime.timezone.utc)
        window_start = now - datetime.timedelta(days=FORECAST_LOOKBACK_DAYS)
        window_end = now + datetime.timedelta(days=FORECAST_LOOKBACK_DAYS)

        table = get_iceberg_table("forecast_predictions")
        df = table.scan(
            row_filter=_between("forecast_timestamp", window_start, window_end)
        ).to_arrow().to_pandas()

        if df.empty:
            return {"status": f"no forecasts in the last {FORECAST_LOOKBACK_DAYS} days"}

        latest_run = df['created_at'].max()
        latest_df = df[df['created_at'] == latest_run].copy()

        # DE-NORMALIZE the target variable using dynamically loaded parameters
        latest_df['target_variable'] = latest_df['predicted_features'].apply(lambda x: denormalize(x[0]))

        pivot_df = latest_df.pivot(index='forecast_timestamp', columns='model_name', values='target_variable').reset_index()
        pivot_df['forecast_timestamp'] = pivot_df['forecast_timestamp'].astype(str)

        return pivot_df.to_dict(orient='records')
    except Exception:
        # Log the detail, return a generic message: the raw exception can carry bucket
        # names and endpoints, and a 200 would let clients read a failure as success.
        logger.exception("forecast/latest failed")
        raise HTTPException(status_code=503, detail="Could not read forecasts from the lakehouse.")


@app.get("/api/v1/metrics/residuals")
def get_residuals():
    try:
        refresh_scaling_parameters()
        now = datetime.datetime.now(datetime.timezone.utc)
        window_start = now - datetime.timedelta(days=RESIDUAL_LOOKBACK_DAYS)

        preds_df = get_iceberg_table("forecast_predictions").scan(
            row_filter=_between("forecast_timestamp", window_start, now)
        ).to_arrow().to_pandas()

        if preds_df.empty:
            return {"status": "insufficient data"}

        # One row per (hour, model, version). Re-runs used to append an identical copy
        # of the same forecast; keeping them would only inflate the sample count.
        preds_df = preds_df.drop_duplicates(
            subset=['forecast_timestamp', 'model_name', 'model_version'], keep='last'
        )

        # Only the actuals that can possibly match those predictions.
        actuals_df = get_iceberg_table("ml_features").scan(
            row_filter=_between("timestamp", preds_df['forecast_timestamp'].min(),
                                preds_df['forecast_timestamp'].max())
        ).to_arrow().to_pandas()

        if actuals_df.empty:
            # Normal early state: the forecasts are for hours that have not happened yet.
            return {"status": "Waiting for actuals to overlap with predictions."}

        # DE-NORMALIZE the features before calculating errors
        preds_df['pred_target'] = preds_df['predicted_features'].apply(lambda x: denormalize(x[0]))
        actuals_df['actual_target'] = actuals_df['features'].apply(lambda x: denormalize(x[0]))

        # STRIP TIMEZONES to prevent Pandas Merge Crashes
        preds_df['forecast_timestamp'] = pd.to_datetime(preds_df['forecast_timestamp']).dt.tz_localize(None)
        actuals_df['timestamp'] = pd.to_datetime(actuals_df['timestamp']).dt.tz_localize(None)

        merged = pd.merge(
            preds_df, actuals_df,
            left_on='forecast_timestamp', right_on='timestamp', how='inner'
        )

        # Safe overlap check
        if merged.empty:
            return {"status": "Waiting for actuals to overlap with predictions."}

        merged['abs_error'] = (merged['actual_target'] - merged['pred_target']).abs()
        merged['sq_error'] = (merged['actual_target'] - merged['pred_target']) ** 2

        # Grouped by version as well as name: the same model_name can hold forecasts
        # from an older champion, and averaging those together reports the error of a
        # model that is no longer serving.
        metrics = merged.groupby(['model_name', 'model_version']).agg(
            mae=('abs_error', 'mean'),
            rmse=('sq_error', lambda x: math.sqrt(x.mean())),
            samples=('abs_error', 'size'),
        ).reset_index()

        return metrics.to_dict(orient='records')
    except Exception:
        logger.exception("metrics/residuals failed")
        raise HTTPException(status_code=503, detail="Could not compute residuals from the lakehouse.")
