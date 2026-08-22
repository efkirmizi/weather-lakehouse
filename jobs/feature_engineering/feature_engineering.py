import logging
import math
import os
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

# The Iceberg and Nessie JARs are baked into $SPARK_HOME/jars by the Dockerfile.
# Do not reintroduce a spark.jars.packages config: it makes Spark run an Ivy
# resolution and re-download every JAR on each run regardless of what is already
# on the classpath - ~200MB per container, plus a hard dependency on Maven
# Central being reachable. Bump the versions in the Dockerfile instead.
SOURCE_TABLE = "nessie.weather.observations"
TARGET_TABLE = "nessie.weather.ml_features"
SCALING_TABLE = "nessie.weather.scaling_parameters"

# THE feature vector, in order, declared once. Five functions in this file used to
# each carry a piece of this knowledge; with three kinds of channel that scatters
# badly, and the layout is what README calls load-bearing.
#
# Channels 0-3 keep their identity and position forever: the serving API reads
# channel 0 as temperature from an image with no copy of this layout, and
# drift_monitor/evaluate_and_promote read it through TEMPERATURE_CHANNEL, which
# only agrees with this file by convention.
SCALED_COLUMNS = [
    # (source column, channel name)
    ("temperature_c",           "temperature"),
    ("humidity_percent",        "humidity"),
    ("precipitation_mm",        "precipitation"),
    ("wind_speed_kmh",          "wind_speed"),
    ("pressure_msl_hpa",        "pressure"),
    ("dew_point_c",             "dew_point"),
    ("cloud_cover_percent",     "cloud_cover"),
    ("shortwave_radiation_wm2", "shortwave_radiation"),
    ("soil_temperature_c",      "soil_temperature"),
    ("soil_moisture_m3m3",      "soil_moisture"),
]

# Emitted as sin/cos pairs, never standardized: a compass bearing and a clock are
# circular, so on a linear scale 359 and 1 sit at opposite extremes and 23:00 looks
# maximally far from midnight. Each contributes two channels, already in [-1, 1].
CYCLICAL_CHANNELS = [
    # (name prefix, period, raw value)
    ("wind_dir", 360.0,  lambda: F.coalesce(F.col("wind_direction_deg"), F.lit(0.0))),
    ("hour",      24.0,  lambda: F.hour("timestamp")),
    ("doy",      365.25, lambda: F.dayofyear("timestamp")),
]

FEATURE_COLS = [column for column, _ in SCALED_COLUMNS]
SERVING_FEATURE_NAMES = (
    [name for _, name in SCALED_COLUMNS]
    + [f"{name}_{part}" for name, _, _ in CYCLICAL_CHANNELS for part in ("sin", "cos")]
)


def cyclical_pair(value, period):
    """(sin, cos) of a value on a circle of the given period.

    Plain Python beside the Spark version so the wrap-around is testable without a
    session; both use the same formula. No production path calls this - it is the
    oracle the local-Spark test in test_feature_engineering.py compares
    standardize()'s channels 10-15 against, which is what actually exercises the
    Spark expression below instead of this twin.
    """
    angle = 2.0 * math.pi * float(value) / period
    return math.sin(angle), math.cos(angle)

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
        # F.hour/F.dayofyear (the calendar channels below) resolve through this
        # setting, not a fixed calendar - they are the only values in the vector
        # whose *content*, not just its formatting, depends on container locale.
        # Pinned so they stay correct if an image ever sets TZ to something other
        # than UTC; today they are UTC only because nothing happens to set it.
        .config("spark.sql.session.timeZone", "UTC")
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
        return {
            column: by_name[name] for column, name in SCALED_COLUMNS
        }
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
    """Imputes nulls with the feature mean, standardizes, and appends the cyclical
    channels, all in one projection.

    Applying the constants directly is what makes an incremental run possible: new
    rows can be normalized with exactly the parameters the existing rows already use.
    """
    columns = []
    for c in FEATURE_COLS:
        mean, std = stats[c]
        filled = F.coalesce(F.col(c), F.lit(mean))
        columns.append(((filled - F.lit(mean)) / F.lit(std)).cast("float"))

    for _, period, value_fn in CYCLICAL_CHANNELS:
        angle = (2.0 * math.pi / period) * value_fn()
        columns.append(F.sin(angle).cast("float"))
        columns.append(F.cos(angle).cast("float"))

    return df.select(F.col("timestamp"), F.array(*columns).alias("features"))


def scaling_parameter_rows(stats: dict) -> list:
    """One (feature_name, feature_index, mean, std) row per published channel.

    Pulled out of write_scaling_parameters so the row layout - the thing an index
    slip would corrupt while every row count and column name still looked right -
    can be asserted on without a Spark session. Standardized columns first, in
    SCALED_COLUMNS order, then the cyclical tail with identity scaling.
    """
    rows = [
        (name, index, stats[column][0], stats[column][1])
        for index, (column, name) in enumerate(SCALED_COLUMNS)
    ]
    rows += [
        (name, len(SCALED_COLUMNS) + offset, 0.0, 1.0)
        for offset, name in enumerate(SERVING_FEATURE_NAMES[len(SCALED_COLUMNS):])
    ]
    return rows


def write_scaling_parameters(spark: SparkSession, stats: dict) -> None:
    """Publishes the inverse transform for the serving layer.

    Only ever called on a full rebuild: on an incremental run the existing rows were
    normalized with the published values, so replacing them would silently make the
    stored vectors and the de-normalization disagree.

    The cyclical channels are published too, with identity scaling. They need no
    inverse transform, but a consumer reading this table must not have to know which
    channels it silently omits.
    """
    rows = scaling_parameter_rows(stats)
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
