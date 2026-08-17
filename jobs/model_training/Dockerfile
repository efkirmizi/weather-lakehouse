# Use the official PyTorch image
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

# 1. Upgrade pip first
RUN pip install --upgrade pip

# 2. Install base data tools
RUN pip install pyarrow pandas numpy

# 3. Install Iceberg separately
RUN pip install pyiceberg[s3fs]

# 4. Install MLflow and Boto3 last
RUN pip install mlflow boto3

COPY . ./

# FORCE SINGLE-THREADED C++ EXTENSIONS TO PREVENT IMPORT DEADLOCKS
ENV PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Create a non-root user and grant ownership
RUN useradd -m ml_user && \
    chown -R ml_user:ml_user /app

USER ml_user