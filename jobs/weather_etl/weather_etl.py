import logging
import os
import time
from typing import Dict, Any, List, Optional, Tuple

import openmeteo_requests
import pandas as pd
import requests_cache
from pyspark.sql import SparkSession
from retry_requests import retry

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("WeatherLakehouse")

# Configuration Constants
# The Iceberg and Nessie JARs are baked into $SPARK_HOME/jars by the Dockerfile.
# Do not reintroduce a spark.jars.packages config: it makes Spark run an Ivy
# resolution and re-download every JAR on each run regardless of what is already
# on the classpath - ~200MB per container, plus a hard dependency on Maven
# Central being reachable. Bump the versions in the Dockerfile instead.
# The Historical API uses a different endpoint than the forecast API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
TABLE_NAME = "nessie.weather.observations"
# Normalization statistics live with the job that actually fits the scaler
# (feature_engineering.py). Recomputing them here would drift from it, not least
# because scaling happens after imputation.
EARLIEST_DATA_DATE = "1940-01-01"

# One ordered declaration of what we pull, what we call it, and what counts as
# physically possible. The Open-Meteo SDK hands variables back positionally -
# hourly.Variables(i), in request order - so the request list and the column list
# have to be generated from the same source. Hand-indexing four was survivable;
# hand-indexing eleven means a reordering silently swaps two columns and every
# downstream number stays plausible.
HOURLY_VARIABLES = [
    # (Open-Meteo name,          our column,                (low, high))
    ("temperature_2m",            "temperature_c",           (-60.0, 60.0)),
    ("relative_humidity_2m",      "humidity_percent",        (0.0, 100.0)),
    ("precipitation",             "precipitation_mm",        (0.0, 500.0)),
    ("wind_speed_10m",            "wind_speed_kmh",          (0.0, 300.0)),
    ("pressure_msl",              "pressure_msl_hpa",        (870.0, 1085.0)),
    ("dew_point_2m",              "dew_point_c",             (-60.0, 60.0)),
    ("cloud_cover",               "cloud_cover_percent",     (0.0, 100.0)),
    ("shortwave_radiation",       "shortwave_radiation_wm2", (0.0, 1400.0)),
    ("soil_temperature_0_to_7cm", "soil_temperature_c",      (-60.0, 70.0)),
    ("soil_moisture_0_to_7cm",    "soil_moisture_m3m3",      (0.0, 1.0)),
    ("wind_direction_10m",        "wind_direction_deg",      (0.0, 360.0)),
]

# ERA5 puts the dew point a hair above the temperature at saturation.
DEW_POINT_TOLERANCE_C = 0.5

DEFAULT_PARAMS = {
    "latitude": 40.98,
    "longitude": 27.51,
    "hourly": [api_name for api_name, _, _ in HOURLY_VARIABLES],
}

def create_spark_session() -> SparkSession:
    """Initializes and returns an Apache Spark session configured with Iceberg and Nessie."""
    logger.info("Initializing SparkSession...")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    return (
        SparkSession.builder
        .appName("WeatherLakehouseHistorical")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
                "org.projectnessie.spark.extensions.NessieSparkSessionExtensions")
        .config("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.nessie.catalog-impl", "org.apache.iceberg.nessie.NessieCatalog")
        .config("spark.sql.catalog.nessie.uri", "http://nessie:19120/api/v2")
        .config("spark.sql.catalog.nessie.ref", "main")
        .config("spark.sql.catalog.nessie.warehouse", "s3://warehouse/")
        .config("spark.sql.catalog.nessie.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
        .config("spark.sql.catalog.nessie.s3.endpoint", "http://minio:9000")
        .config("spark.sql.catalog.nessie.s3.path-style-access", "true")
        .config("spark.sql.catalog.nessie.s3.access-key-id", access_key)
        .config("spark.sql.catalog.nessie.s3.secret-access-key", secret_key)
        .config("spark.sql.catalog.nessie.client.region", "us-east-1")
        .config("spark.sql.catalog.nessie.client-api-version", "2")
        .getOrCreate()
    )

def get_watermark(spark: SparkSession) -> Optional[pd.Timestamp]:
    """Newest timestamp already in the table, or None when there is nothing yet.

    None rather than EARLIEST_DATA_DATE so callers can tell "empty table" from "we
    genuinely hold the 1940-01-01 00:00 row" - drop_already_ingested would otherwise
    discard that first hour on the very first run.
    """
    catalog, database, _ = TABLE_NAME.split(".")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{database}")
    
    if not spark.catalog.tableExists(TABLE_NAME):
        logger.info(f"Table {TABLE_NAME} does not exist. Starting from {EARLIEST_DATA_DATE}.")
        return None
    
    try:
        max_date_row = spark.sql(f"SELECT MAX(timestamp) as max_ts FROM {TABLE_NAME}").collect()[0]
        if max_date_row['max_ts'] is not None:
            latest_ts = pd.to_datetime(max_date_row['max_ts'], utc=True)
            logger.info(f"Found existing data. Latest timestamp is {latest_ts}.")
            return latest_ts
    except Exception as e:
        logger.warning(f"Could not read max timestamp, starting from {EARLIEST_DATA_DATE}. Reason: {e}")

    return None

def generate_date_chunks(watermark: Optional[pd.Timestamp]) -> List[Tuple[str, str]]:
    """Missing date ranges, in 5-year increments to prevent API timeouts.

    A watermark short of 23:00 re-fetches its own day, because the archive API only
    serves whole days. drop_already_ingested is what stops that becoming duplicates.
    """
    if watermark is None:
        start_date = pd.to_datetime(EARLIEST_DATA_DATE, utc=True).date()
    else:
        start_date = (watermark + pd.Timedelta(hours=1)).date()
    # Open-Meteo Historical API is safely available up to "yesterday"
    end_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    
    if start_date > end_date:
        return []

    chunks = []
    curr = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    while curr <= end:
        # Advance by 5 years, minus 1 day to prevent overlap
        next_date = curr + pd.DateOffset(years=5) - pd.Timedelta(days=1)
        if next_date > end:
            next_date = end
            
        chunks.append((curr.strftime("%Y-%m-%d"), next_date.strftime("%Y-%m-%d")))
        curr = next_date + pd.Timedelta(days=1)
        
    return chunks

def extract_weather_chunk(url: str, params: Dict[str, Any], start_str: str, end_str: str) -> Any:
    """Extracts a specific date range of hourly weather forecast data."""
    logger.info(f"Extracting historical data from {start_str} to {end_str}...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    
    chunk_params = params.copy()
    chunk_params["start_date"] = start_str
    chunk_params["end_date"] = end_str
    
    responses = client.weather_api(url, params=chunk_params)
    return responses[0]

def transform_weather_data(response: Any) -> pd.DataFrame:
    """Transforms raw API response into a structured pandas DataFrame."""
    hourly = response.Hourly()
    hourly_data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
    }
    # Positional, in the order they were requested - which is why both come from
    # HOURLY_VARIABLES rather than being written out twice.
    for index, (_, column, _) in enumerate(HOURLY_VARIABLES):
        hourly_data[column] = hourly.Variables(index).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    # Drop any null rows just in case the API returned gaps for very old dates.
    # The first 7 hours of 1940-01-01 are the known case: precipitation and
    # shortwave_radiation need a preceding accumulation interval.
    df.dropna(inplace=True)
    return df

def drop_already_ingested(df: pd.DataFrame, watermark: Optional[pd.Timestamp]) -> pd.DataFrame:
    """Removes rows the table already holds.

    The archive API only serves whole days, so a chunk starting from a mid-day
    watermark comes back carrying hours that are already ingested. That happens
    whenever transform_weather_data drops a NaN tail and leaves the watermark short of
    23:00. load_to_lakehouse is a plain append with no primary key, so re-appending
    them duplicates rows outright: the feature job's row-count invariant then forces a
    full rebuild, and every training window spanning a duplicated hour is thrown away
    as a discontinuity.
    """
    if watermark is None:
        return df
    return df[df["timestamp"] > watermark]


def run_data_quality_checks(df: pd.DataFrame) -> None:
    """Runs strict data quality assertions before Lakehouse ingestion."""
    logger.info("Executing Data Quality Gates...")
    
    # 1. Null Checks
    if df.isnull().values.any():
        raise ValueError("Data Quality Failure: DataFrame contains NULL values.")
        
    # 2. Physical Range Bounds, from the same table that defines the columns.
    for _, column, (low, high) in HOURLY_VARIABLES:
        if not df[column].between(low, high).all():
            raise ValueError(
                f"Data Quality Failure: {column} outside its physical range "
                f"[{low}, {high}]."
            )

    # 2b. A cross-column invariant, which is the only kind that catches a column
    # mix-up: both values stay individually plausible when two columns are swapped.
    # Air cannot hold more moisture than saturation, so the dew point can never
    # exceed the temperature.
    excess = df["dew_point_c"] - df["temperature_c"]
    if (excess > DEW_POINT_TOLERANCE_C).any():
        raise ValueError(
            f"Data Quality Failure: dew point exceeds temperature by up to "
            f"{excess.max():.2f} C. The columns are probably misaligned."
        )

    # 3. Sequential Continuity Check
    #
    # A warning rather than a failure on purpose: the Open-Meteo archive genuinely
    # has holes in the older decades, and transform_weather_data drops those rows.
    # Downstream is what has to be gap-safe - IcebergTimeSeriesDataset skips windows
    # that span a gap, and batch_inference refuses a discontinuous context window.
    df_sorted = df.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff().dropna()
    gaps = time_diffs[time_diffs != pd.Timedelta(hours=1)]
    if not gaps.empty:
        logger.warning(
            f"Data Quality Warning: {len(gaps)} time series gap(s) detected, "
            f"largest {gaps.max()}. Windows spanning them will be skipped downstream."
        )
        
    logger.info("Data Quality Gates Passed! Data is pristine.")

def load_to_lakehouse(spark: SparkSession, pandas_df: pd.DataFrame) -> None:
    """Appends data to the Iceberg table."""
    spark_df = spark.createDataFrame(pandas_df)
    spark_df.write \
        .format("iceberg") \
        .mode("append") \
        .saveAsTable(TABLE_NAME)

def main() -> None:
    spark = None
    try:
        spark = create_spark_session()
        
        watermark = get_watermark(spark)
        date_chunks = generate_date_chunks(watermark)
        
        if not date_chunks:
            logger.info("Lakehouse is already up to date! No missing data to fetch.")
            return

        logger.info(f"Identified {len(date_chunks)} chunks of missing data to fetch.")

        for start_str, end_str in date_chunks:
            # 1. EXTRACT
            response = extract_weather_chunk(API_URL, DEFAULT_PARAMS, start_str, end_str)

            # 2a. TRANSFORM
            pandas_df = transform_weather_data(response)

            # 2b. DROP WHAT WE ALREADY HAVE
            pandas_df = drop_already_ingested(pandas_df, watermark)
            if pandas_df.empty:
                logger.info(
                    f"Chunk {start_str} -> {end_str} holds nothing newer than {watermark}. Skipping."
                )
                continue

            # 2c. DATA QUALITY GATE
            run_data_quality_checks(pandas_df)

            # 3. LOAD
            load_to_lakehouse(spark, pandas_df)
            # Advance so the next chunk is measured against what was just committed,
            # not against the watermark this run started with.
            watermark = pandas_df["timestamp"].max()
            logger.info(f"Successfully committed chunk {start_str} -> {end_str} to Iceberg.")

            # ADD THIS SLEEP: Pause for seconds to respect Open-Meteo's rate limits
            time.sleep(10)

        logger.info("Historical backfill completed successfully!")
    except Exception as e:
        logger.critical(f"ETL Pipeline failed: {e}", exc_info=True)
        raise
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()
