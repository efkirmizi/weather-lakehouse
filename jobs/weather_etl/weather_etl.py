import logging
import os
import time
from typing import Dict, Any, List, Tuple

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
ICEBERG_VERSION = "1.5.0"
NESSIE_VERSION = "0.108.4"
# The Historical API uses a different endpoint than the forecast API
API_URL = "https://archive-api.open-meteo.com/v1/archive"
TABLE_NAME = "nessie.weather.observations"
# Normalization statistics live with the job that actually fits the scaler
# (feature_engineering.py). Recomputing them here would drift from it, not least
# because scaling happens after imputation.
EARLIEST_DATA_DATE = "1940-01-01"

DEFAULT_PARAMS = {
    "latitude": 40.98,
    "longitude": 27.51,
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
}

def create_spark_session() -> SparkSession:
    """Initializes and returns an Apache Spark session configured with Iceberg and Nessie."""
    logger.info("Initializing SparkSession...")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    return (
        SparkSession.builder
        .appName("WeatherLakehouseHistorical")
        .config("spark.jars.packages", ",".join([
            f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION}",
            f"org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}",
            f"org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:{NESSIE_VERSION}",
        ]))
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

def get_watermark(spark: SparkSession) -> pd.Timestamp:
    """Queries Iceberg to find the latest timestamp we have data for."""
    catalog, database, _ = TABLE_NAME.split(".")
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {catalog}.{database}")
    
    table_exists = spark.catalog.tableExists(TABLE_NAME)
    
    if not table_exists:
        logger.info(f"Table {TABLE_NAME} does not exist. Starting from {EARLIEST_DATA_DATE}.")
        return pd.to_datetime(EARLIEST_DATA_DATE, utc=True)
    
    try:
        max_date_row = spark.sql(f"SELECT MAX(timestamp) as max_ts FROM {TABLE_NAME}").collect()[0]
        if max_date_row['max_ts'] is not None:
            latest_ts = pd.to_datetime(max_date_row['max_ts'], utc=True)
            logger.info(f"Found existing data. Latest timestamp is {latest_ts}.")
            return latest_ts
    except Exception as e:
        logger.warning(f"Could not read max timestamp, defaulting to {EARLIEST_DATA_DATE}. Reason: {e}")
        
    return pd.to_datetime(EARLIEST_DATA_DATE, utc=True)

def generate_date_chunks(start_ts: pd.Timestamp) -> List[Tuple[str, str]]:
    """Calculates missing date ranges and chunks them into 5-year increments to prevent API timeouts."""
    start_date = (start_ts + pd.Timedelta(hours=1)).date()
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
        "temperature_c": hourly.Variables(0).ValuesAsNumpy(),
        "humidity_percent": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation_mm": hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed_kmh": hourly.Variables(3).ValuesAsNumpy()
    }
    
    df = pd.DataFrame(data=hourly_data)
    # Drop any null rows just in case the API returned gaps for very old dates
    df.dropna(inplace=True)
    return df

def run_data_quality_checks(df: pd.DataFrame) -> None:
    """Runs strict data quality assertions before Lakehouse ingestion."""
    logger.info("Executing Data Quality Gates...")
    
    # 1. Null Checks
    if df.isnull().values.any():
        raise ValueError("Data Quality Failure: DataFrame contains NULL values.")
        
    # 2. Physical Range Bounds (e.g., Temperature between -60C and 60C)
    if not df['temperature_c'].between(-60, 60).all():
        raise ValueError("Data Quality Failure: Temperature out of physical bounds.")
        
    if not df['humidity_percent'].between(0, 100).all():
        raise ValueError("Data Quality Failure: Humidity out of bounds (0% to 100%).")
        
    if not df['wind_speed_kmh'].between(0, 300).all(): # Max realistic wind speed
        raise ValueError("Data Quality Failure: Wind speed exceeds realistic bounds.")
        
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
            
            # 2b. DATA QUALITY GATE
            run_data_quality_checks(pandas_df)
            
            # 3. LOAD
            load_to_lakehouse(spark, pandas_df)
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
