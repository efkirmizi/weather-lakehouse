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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        bash \
        procps \
    && rm -rf /var/lib/apt/lists/*

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
# Install Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy application script
# -----------------------------
COPY --chown=spark:spark feature_engineering.py .

USER spark

# Airflow overrides this command, but it serves as a safe default fallback
CMD ["python3", "feature_engineering.py"]