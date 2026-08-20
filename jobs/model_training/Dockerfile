# Use the official PyTorch image
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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