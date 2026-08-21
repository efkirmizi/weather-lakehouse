import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from pyiceberg.expressions import And, GreaterThanOrEqual, LessThanOrEqual, Or

from lakehouse import load_iceberg_catalog

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Drift_Monitor")

# Configuration
P_VALUE_THRESHOLD = float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05"))
# The baseline is drawn from the same time of year in previous years, +/- this many
# days. Comparing the last 24h against "the previous 30 days" on a strongly seasonal
# hourly series flags every ordinary seasonal transition as drift.
BASELINE_DOY_WINDOW_DAYS = int(os.getenv("DRIFT_BASELINE_DOY_WINDOW_DAYS", "7"))
# How many years of climatology to draw on. 30 is the WMO climate-normal convention,
# and it also bounds the read: without it this job pulled the entire feature table
# (~760k rows) to look at a few thousand samples.
BASELINE_YEARS = int(os.getenv("DRIFT_BASELINE_YEARS", "30"))
# Below this the KS test is not worth acting on.
MIN_BASELINE_SAMPLES = int(os.getenv("DRIFT_MIN_BASELINE_SAMPLES", "336"))
# Do not retrain a DAG that already started within this window. Each trigger is a
# from-scratch run of a full model; without a cooldown a warm spell retrains daily.
COOLDOWN_HOURS = float(os.getenv("DRIFT_COOLDOWN_HOURS", "168"))
# How far back to look for the newest row before falling back to a full scan. 01 runs
# daily, so on any healthy stack the anchor is found in a few thousand rows.
ANCHOR_PROBE_DAYS = int(os.getenv("DRIFT_ANCHOR_PROBE_DAYS", "30"))
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
# Every model behind the pipeline should react to drift, not just the Transformer.
RETRAIN_DAG_IDS = [
    dag_id.strip()
    for dag_id in os.getenv(
        "DRIFT_RETRAIN_DAGS", "03a_train_pytorch_lstm,03b_train_pytorch_transformer"
    ).split(",")
    if dag_id.strip()
]


def _airflow_credentials():
    """Reads the Airflow API credentials from the environment.

    Deliberately no fallback: a wrong hardcoded password fails as a confusing 401
    at the worst possible moment, and a right one is a secret in source control.
    """
    user = os.getenv("AIRFLOW_API_USER")
    password = os.getenv("AIRFLOW_API_PASSWORD")
    if not user or not password:
        return None
    return (user, password)


def _recently_triggered(dag_id: str, auth) -> bool:
    """Whether this DAG already started a run inside the cooldown window."""
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=COOLDOWN_HOURS)
    try:
        response = requests.get(
            f"{AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns",
            params={"start_date_gte": since.isoformat(), "limit": 1},
            auth=auth,
            timeout=30,
        )
        if not response.ok:
            logger.warning(f"Could not check cooldown for '{dag_id}' (HTTP {response.status_code}).")
            return False
        return response.json().get("total_entries", 0) > 0
    except Exception as e:
        # Failing to retrain on real drift is worse than an extra run, so proceed.
        logger.warning(f"Cooldown check failed for '{dag_id}' ({e}); triggering anyway.")
        return False


def trigger_retraining():
    """Triggers the configured model retraining DAGs via Airflow's REST API."""
    auth = _airflow_credentials()
    if auth is None:
        logger.error(
            "AIRFLOW_API_USER / AIRFLOW_API_PASSWORD are not set. "
            "Drift was detected but retraining could not be triggered."
        )
        return

    payload = {
        # The branch operator in 03a/03b honours this: concept drift means the old
        # weights are the problem, so warm-starting from them defeats the purpose.
        "conf": {"TRAINING_MODE": "SCRATCH"},
        "note": "Triggered automatically by KS-Test Drift Detection",
    }

    for dag_id in RETRAIN_DAG_IDS:
        if _recently_triggered(dag_id, auth):
            logger.info(
                f"'{dag_id}' already ran within the last {COOLDOWN_HOURS:.0f}h. "
                "Skipping to avoid a retraining loop."
            )
            continue

        url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns"
        logger.info(f"Triggering full retraining for '{dag_id}'...")
        try:
            response = requests.post(url, json=payload, auth=auth, timeout=30)
            if response.status_code == 200:
                logger.info(f"Successfully triggered '{dag_id}'.")
            else:
                logger.error(f"Failed to trigger '{dag_id}'. Status: {response.status_code}, Response: {response.text}")
        except Exception as e:
            logger.error(f"API connection failed for '{dag_id}': {e}")


def latest_timestamp(table):
    """Newest timestamp in the feature table, or None if it holds nothing.

    Probes a recent window before reading everything. Scanning the whole timestamp
    column - which is what this did - is the same unbounded read that climatology_filter
    exists to avoid, only one column wide.
    """
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=ANCHOR_PROBE_DAYS)
    probe = table.scan(
        row_filter=GreaterThanOrEqual("timestamp", since.isoformat()),
        selected_fields=("timestamp",),
    ).to_arrow()
    if probe.num_rows:
        return pd.to_datetime(probe.column("timestamp").to_pandas().max(), utc=True)

    logger.warning(
        f"No feature rows in the last {ANCHOR_PROBE_DAYS} days - 01 may have stopped. "
        "Falling back to a full scan to find the anchor."
    )
    everything = table.scan(selected_fields=("timestamp",)).to_arrow()
    if everything.num_rows == 0:
        return None
    return pd.to_datetime(everything.column("timestamp").to_pandas().max(), utc=True)


def climatology_filter(anchor: pd.Timestamp):
    """Row filter covering this time of year across the last BASELINE_YEARS years.

    ml_features is written sorted by timestamp, so these narrow ranges prune most of
    the Parquet row groups instead of reading the whole table.
    """
    clauses = []
    for offset in range(BASELINE_YEARS + 1):
        centre = anchor - pd.DateOffset(years=offset)
        start = centre - pd.Timedelta(days=BASELINE_DOY_WINDOW_DAYS)
        end = centre + pd.Timedelta(days=BASELINE_DOY_WINDOW_DAYS)
        clauses.append(And(
            GreaterThanOrEqual("timestamp", start.isoformat()),
            LessThanOrEqual("timestamp", end.isoformat()),
        ))
    return clauses[0] if len(clauses) == 1 else Or(*clauses)


def main():
    logger.info("Connecting to Lakehouse to analyze feature distributions...")
    catalog = load_iceberg_catalog()

    table = catalog.load_table(("weather", "ml_features"))

    # The newest timestamp anchors the seasonal window; read it first so the main
    # scan can be bounded instead of pulling the entire table.
    anchor = latest_timestamp(table)
    if anchor is None:
        logger.info("Feature table is empty. Skipping.")
        sys.exit(0)

    df = table.scan(row_filter=climatology_filter(anchor)).to_arrow().to_pandas()
    if df.empty:
        logger.info("No data in the seasonal window. Skipping.")
        sys.exit(0)

    # Convert timestamps and extract the raw temperature feature (Index 0)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['temperature'] = df['features'].apply(lambda x: x[0])
    df = df.sort_values('timestamp')

    # 2. Target = last 24 hours. Baseline = the same time of year in earlier years.
    latest_time = df['timestamp'].max()
    cutoff_24h = latest_time - pd.Timedelta(hours=24)

    target_df = df[df['timestamp'] > cutoff_24h]
    history = df[df['timestamp'] <= cutoff_24h]

    target_doy = latest_time.dayofyear
    day_offset = (history['timestamp'].dt.dayofyear - target_doy).abs()
    # Circular, so late December and early January are neighbours.
    seasonal_distance = np.minimum(day_offset, 365 - day_offset)
    baseline_df = history[seasonal_distance <= BASELINE_DOY_WINDOW_DAYS]

    if target_df.empty or len(baseline_df) < MIN_BASELINE_SAMPLES:
        logger.info(
            f"Not enough history for a seasonal comparison "
            f"({len(baseline_df)} baseline samples, {len(target_df)} target hours). Skipping."
        )
        sys.exit(0)

    logger.info(
        f"Comparing {len(target_df)} hours since {cutoff_24h} against {len(baseline_df)} "
        f"climatological samples within +/-{BASELINE_DOY_WINDOW_DAYS} days of day {target_doy}, "
        f"over the last {BASELINE_YEARS} years."
    )

    # 3. Perform Kolmogorov-Smirnov Test
    stat, p_value = ks_2samp(baseline_df['temperature'], target_df['temperature'])

    logger.info(f"Kolmogorov-Smirnov Test Results -> Statistic: {stat:.4f}, P-Value: {p_value:.4f}")

    # 4. Evaluate Drift
    if p_value < P_VALUE_THRESHOLD:
        logger.warning(f"SIGNIFICANT DATA DRIFT DETECTED! (p-value {p_value:.4f} < {P_VALUE_THRESHOLD})")
        trigger_retraining()
    else:
        logger.info("Distributions are stable. No drift detected.")


if __name__ == "__main__":
    main()
