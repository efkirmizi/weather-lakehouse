FROM python:3.11-slim-bookworm

# -----------------------------
# Environment
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark \
    PATH=/usr/lib/jvm/java-17-openjdk-amd64/bin:/usr/local/bin:$PATH \
    HOME=/home/spark

# -----------------------------
# Install Java & System Utilities
# -----------------------------
# ADDED: wget so we can download the JARs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        wget \
        bash \
        procps \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python dependencies (Moved up!)
# -----------------------------
# PySpark MUST be installed before we can put JARs in its directory
RUN pip install \
    pyspark==3.5.2 \
    pandas \
    openmeteo-requests \
    requests-cache \
    retry-requests

# -----------------------------
# Bake Maven JARs directly into Spark
# -----------------------------
RUN wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.5.0/iceberg-spark-runtime-3.5_2.12-1.5.0.jar -P $SPARK_HOME/jars/ && \
    wget https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws-bundle/1.5.0/iceberg-aws-bundle-1.5.0.jar -P $SPARK_HOME/jars/ && \
    wget https://repo1.maven.org/maven2/org/projectnessie/nessie-integrations/nessie-spark-extensions-3.5_2.12/0.108.4/nessie-spark-extensions-3.5_2.12-0.108.4.jar -P $SPARK_HOME/jars/

# -----------------------------
# Create spark user & group
# -----------------------------
RUN groupadd --system spark && \
    useradd --system \
        --gid spark \
        --create-home \
        --home-dir /home/spark \
        spark

# -----------------------------
# Create directories Spark/Ivy needs
# -----------------------------
RUN mkdir -p \
        /home/spark/.ivy2/cache \
        /home/spark/.ivy2/jars \
        /opt/spark/work-dir \
    && chown -R spark:spark \
        /home/spark \
        /opt/spark

WORKDIR /opt/spark/work-dir

# -----------------------------
# Copy application script
# -----------------------------
COPY --chown=spark:spark weather_etl.py .

USER spark

# Airflow overrides this command, but it serves as a safe default fallback
CMD ["python3", "weather_etl.py"]