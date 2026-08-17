import logging
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler, VectorAssembler, Imputer
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEngineering")

ICEBERG_VERSION = "1.5.0"
NESSIE_VERSION = "0.108.4"
SOURCE_TABLE = "nessie.weather.observations"
TARGET_TABLE = "nessie.weather.ml_features"

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

def main():
    spark = None
    try:
        spark = create_spark_session()
        
        # 1. EXTRACT RAW DATA
        logger.info(f"Reading raw data from {SOURCE_TABLE}...")
        spark_df = spark.table(SOURCE_TABLE).orderBy("timestamp")
        
        feature_cols = ["temperature_c", "humidity_percent", "precipitation_mm", "wind_speed_kmh"]
        
        # 2. HANDLE MISSING DATA (Imputation)
        # Deep learning models crash on Nulls. We use PySpark's Imputer to safely handle gaps.
        logger.info("Imputing missing values...")
        imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols).setStrategy("mean")
        imputed_df = imputer.fit(spark_df).transform(spark_df)

        # 3. VECTORIZE
        logger.info("Assembling features into Vector...")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
        vector_df = assembler.transform(imputed_df)

        # 4. SCALE (Standardization)
        logger.info("Scaling features (Mean=0, StdDev=1)...")
        scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_vector", withStd=True, withMean=True)
        scaler_model = scaler.fit(vector_df)
        ml_ready_df = scaler_model.transform(vector_df)

        # 5. CONVERT & LOAD
        # Convert Spark's DenseVector to a standard Array of Floats (PyArrow/PyTorch friendly)
        final_df = ml_ready_df.withColumn(
            "features", 
            vector_to_array(col("scaled_vector"), dtype="float32")
        ).select("timestamp", "features")
        
        logger.info(f"Writing normalized ML features to {TARGET_TABLE}...")
        final_df.write \
            .format("iceberg") \
            .mode("overwrite") \
            .saveAsTable(TARGET_TABLE)
            
        logger.info("Feature Engineering completed successfully.")

    except Exception as e:
        logger.critical(f"Feature Engineering failed: {e}", exc_info=True)
        raise
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    main()