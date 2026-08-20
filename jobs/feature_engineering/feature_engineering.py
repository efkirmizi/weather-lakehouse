import logging
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

ICEBERG_VERSION = "1.5.0"
NESSIE_VERSION = "0.108.4"
SOURCE_TABLE = "nessie.weather.observations"
TARGET_TABLE = "nessie.weather.ml_features"
SCALING_TABLE = "nessie.weather.scaling_parameters"

FEATURE_COLS = ["temperature_c", "humidity_percent", "precipitation_mm", "wind_speed_kmh"]
# Names the serving layer looks these up by; order matches FEATURE_COLS and the
# position of each value inside the ml_features vector.
SERVING_FEATURE_NAMES = ["temperature", "humidity", "precipitation", "wind_speed"]

# How far the global mean/std may move before the whole table has to be rebuilt.
# Expressed as a fraction of the feature's own standard deviation, so a near-zero
# mean like precipitation does not trigger a rebuild on every drizzle.
REBUILD_TOLERANCE = float(os.getenv("FEATURE_REBUILD_TOLERANCE", "0.01"))

def create_spark_session() -> SparkSession:
    """Initializes Spark with Nessie and Iceberg extensions."""
    logger.info("Initializing SparkSession for Feature Engineering...")
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    return (
        SparkSession.builder
        .appName("WeatherML_Feature_Engineering")
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

def compute_global_stats(spark: SparkSession) -> dict:
    """Mean and standard deviation per feature over the whole observations table.

    F.stddev is the sample standard deviation, which is what Spark ML's
    StandardScaler used before this job stopped fitting one - so the numbers stay
    comparable with the parameters already published.
    """
    aggs = []
    for c in FEATURE_COLS:
        aggs += [F.mean(c).alias(f"{c}__mean"), F.stddev(c).alias(f"{c}__std")]
    row = spark.table(SOURCE_TABLE).select(*aggs).collect()[0]

    stats = {}
    for c in FEATURE_COLS:
        mean = float(row[f"{c}__mean"] or 0.0)
        std = float(row[f"{c}__std"] or 0.0)
        if std == 0.0:
            # A constant column would otherwise divide by zero.
            std = 1.0
        stats[c] = (mean, std)
    return stats


def load_published_stats(spark: SparkSession):
    """The parameters the rows already in ml_features were normalized with."""
    if not spark.catalog.tableExists(SCALING_TABLE):
        return None
    by_name = {
        r["feature_name"]: (float(r["mean_value"]), float(r["std_value"]))
        for r in spark.table(SCALING_TABLE).collect()
    }
    try:
        return {col: by_name[name] for col, name in zip(FEATURE_COLS, SERVING_FEATURE_NAMES)}
    except KeyError:
        return None


def stats_drifted(published: dict, current: dict, tolerance: float) -> bool:
    """Whether renormalizing the whole table is worth it.

    Both differences are measured against the published standard deviation, which
    gives every feature the same scale-free yardstick.
    """
    for c in FEATURE_COLS:
        old_mean, old_std = published[c]
        new_mean, new_std = current[c]
        scale = max(abs(old_std), 1e-12)
        if abs(new_mean - old_mean) / scale > tolerance:
            logger.info(f"{c}: mean moved {abs(new_mean - old_mean) / scale:.4f} std - rebuild.")
            return True
        if abs(new_std - old_std) / scale > tolerance:
            logger.info(f"{c}: std moved {abs(new_std - old_std) / scale:.4f} std - rebuild.")
            return True
    return False


def standardize(df: DataFrame, stats: dict) -> DataFrame:
    """Imputes nulls with the feature mean and standardizes, all in one projection.

    This replaces Imputer + VectorAssembler + StandardScaler. Applying the constants
    directly is what makes an incremental run possible: new rows can be normalized
    with exactly the parameters the existing rows already use.
    """
    columns = []
    for c in FEATURE_COLS:
        mean, std = stats[c]
        filled = F.coalesce(F.col(c), F.lit(mean))
        columns.append(((filled - F.lit(mean)) / F.lit(std)).cast("float"))
    return df.select(F.col("timestamp"), F.array(*columns).alias("features"))


def write_scaling_parameters(spark: SparkSession, stats: dict) -> None:
    """Publishes the inverse transform for the serving layer.

    Only ever called on a full rebuild: on an incremental run the existing rows were
    normalized with the published values, so replacing them would silently make the
    stored vectors and the de-normalization disagree.
    """
    rows = [
        (SERVING_FEATURE_NAMES[i], i, stats[c][0], stats[c][1])
        for i, c in enumerate(FEATURE_COLS)
    ]
    for name, _, mean_val, std_val in rows:
        logger.info(f"Scaling parameter -> {name}: mean={mean_val:.4f}, std={std_val:.4f}")

    schema = "feature_name STRING, feature_index INT, mean_value DOUBLE, std_value DOUBLE"
    spark.createDataFrame(rows, schema=schema).write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable(SCALING_TABLE)
    logger.info(f"Wrote {len(rows)} scaling parameters to {SCALING_TABLE}.")


def rebuild_reason(spark: SparkSession, published, current, watermark) -> str:
    """Why this run has to renormalize everything, or empty string to go incremental."""
    if os.getenv("FEATURE_REBUILD", "").strip().lower() == "full":
        return "FEATURE_REBUILD=full was requested"
    if watermark is None:
        return f"{TARGET_TABLE} does not exist yet or is empty"
    if published is None:
        return "no scaling parameters have been published yet"

    # The incremental path only picks up observations *newer* than the watermark, so
    # anything that lands behind it - a backfilled gap, a corrected re-ingest - would
    # be skipped forever. The drift check cannot catch that either: a handful of old
    # rows barely moves a global mean. The invariant that does catch it is that every
    # observation up to the watermark must already have a feature row.
    covered = spark.table(SOURCE_TABLE).where(F.col("timestamp") <= F.lit(watermark)).count()
    present = spark.table(TARGET_TABLE).count()
    if covered != present:
        return (
            f"{TARGET_TABLE} is missing rows at or before the watermark "
            f"({covered} observations up to {watermark}, {present} feature rows)"
        )

    if stats_drifted(published, current, REBUILD_TOLERANCE):
        return f"normalization drifted beyond {REBUILD_TOLERANCE:.1%} of a standard deviation"
    return ""


def main():
    spark = None
    try:
        spark = create_spark_session()

        current_stats = compute_global_stats(spark)
        published_stats = load_published_stats(spark)

        watermark = None
        if spark.catalog.tableExists(TARGET_TABLE):
            watermark = spark.table(TARGET_TABLE).select(F.max("timestamp")).collect()[0][0]

        reason = rebuild_reason(spark, published_stats, current_stats, watermark)

        if reason:
            logger.info(f"FULL rebuild: {reason}.")
            source = spark.table(SOURCE_TABLE)
            stats, mode = current_stats, "overwrite"
        else:
            source = spark.table(SOURCE_TABLE).where(F.col("timestamp") > F.lit(watermark))
            new_rows = source.count()
            if new_rows == 0:
                logger.info(f"INCREMENTAL: nothing newer than {watermark}. Table is up to date.")
                return
            logger.info(f"INCREMENTAL: {new_rows} observations newer than {watermark}.")
            # Deliberately the published parameters, not the freshly computed ones:
            # every row in the table has to share one normalization.
            stats, mode = published_stats, "append"

        logger.info(f"Writing ML features to {TARGET_TABLE} (mode={mode})...")
        standardize(source, stats).orderBy("timestamp").write \
            .format("iceberg") \
            .mode(mode) \
            .saveAsTable(TARGET_TABLE)

        if mode == "overwrite":
            write_scaling_parameters(spark, current_stats)

        logger.info("Feature Engineering completed successfully.")

    except Exception as e:
        logger.critical(f"Feature Engineering failed: {e}", exc_info=True)
        raise
    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    main()
