import os
import sys
import logging
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Lakehouse_Janitor")

# Configuration
# The Iceberg and Nessie JARs are baked into $SPARK_HOME/jars by the Dockerfile.
# Do not reintroduce a spark.jars.packages config: it makes Spark run an Ivy
# resolution and re-download every JAR on each run regardless of what is already
# on the classpath - ~200MB per container, plus a hard dependency on Maven
# Central being reachable. Bump the versions in the Dockerfile instead.
CATALOG = "nessie"

# The tables we want to actively maintain
TABLES_TO_MAINTAIN = [
    "weather.observations",
    "weather.ml_features",
    "weather.forecast_predictions",
    # Tiny, but rewritten in full on every rebuild, so it accumulates snapshots.
    "weather.scaling_parameters",
]

def create_spark_session() -> SparkSession:
    """Initializes Spark session with Iceberg and Nessie Extensions to enable CALL procedures."""
    logger.info("Booting up maintenance SparkSession...")
    
    return (
        SparkSession.builder
        .appName("LakehouseMaintenance")
        # CRITICAL: These extensions enable the CALL system procedures for maintenance
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
        .config("spark.sql.catalog.nessie.s3.access-key-id", os.getenv("AWS_ACCESS_KEY_ID"))
        .config("spark.sql.catalog.nessie.s3.secret-access-key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        # Both are required, exactly as in weather_etl.py and feature_engineering.py.
        # Without the region the AWS SDK walks its provider chain, fails to find one,
        # and every S3FileIO call dies before any maintenance runs.
        .config("spark.sql.catalog.nessie.client.region", "us-east-1")
        .config("spark.sql.catalog.nessie.client-api-version", "2")
        .getOrCreate()
    )

def run_maintenance(spark: SparkSession, table: str) -> bool:
    """Compacts data files and then rewrites the manifests that point at them.

    Snapshot expiry and orphan sweeping are deliberately absent. Nessie sets
    `gc.enabled=false` on the tables it manages, because in a Git-like catalog the
    same data file can be referenced from several branches and tags - deleting it
    from one branch's point of view would corrupt the others. Iceberg honours that
    property and rejects both procedures with:

        Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)

    Reclaiming that storage is `nessie-gc`'s job: it computes the live set across
    every reference before sweeping. See the README.
    """
    full_table_name = f"{CATALOG}.{table}"
    logger.info(f"--- Starting Maintenance for {full_table_name} ---")

    try:
        # 1. COMPACTION: merge small appends into large Parquet files.
        logger.info("1/2 Compacting small data files...")
        spark.sql(f"CALL {CATALOG}.system.rewrite_data_files(table => '{table}')")

        # 2. MANIFEST REWRITE: compaction leaves one manifest per historical commit
        # still pointing into the table. Collapsing them keeps planning cheap.
        logger.info("2/2 Rewriting manifests...")
        spark.sql(f"CALL {CATALOG}.system.rewrite_manifests(table => '{table}')")

        logger.info(f"--- Maintenance Complete for {full_table_name} ---\n")
        return True
    except Exception as e:
        logger.error(f"Maintenance failed on table {full_table_name}: {e}", exc_info=True)
        return False

def tables_for_this_run():
    """One table when the DAG maps a task per table, all of them when run by hand."""
    requested = os.getenv("MAINTENANCE_TABLE", "").strip()
    if not requested:
        return TABLES_TO_MAINTAIN
    if requested not in TABLES_TO_MAINTAIN:
        logger.warning(f"MAINTENANCE_TABLE={requested!r} is not in the maintained set; running it anyway.")
    return [requested]


def main():
    spark = create_spark_session()
    failed = []

    try:
        for table in tables_for_this_run():
            # Check if table exists before running maintenance
            if spark.catalog.tableExists(f"{CATALOG}.{table}"):
                if not run_maintenance(spark, table):
                    failed.append(table)
            else:
                logger.warning(f"Table {table} does not exist yet. Skipping.")
    finally:
        spark.stop()

    if failed:
        # Silently "succeeding" through a broken maintenance window is how storage
        # bloat goes unnoticed for months.
        logger.error(f"Maintenance failed for: {', '.join(failed)}")
        sys.exit(1)

    logger.info("Weekly Lakehouse Maintenance finished successfully.")

if __name__ == "__main__":
    main()