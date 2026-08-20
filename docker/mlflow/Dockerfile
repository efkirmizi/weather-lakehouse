FROM python:3.11-slim

# The tracking server used to pip-install itself on every boot from a bare
# python:3.11-slim. Same reasoning as the Airflow image: pin it, build it once.
RUN pip install --no-cache-dir \
    mlflow==3.15.1 \
    boto3==1.43.74

EXPOSE 5000
