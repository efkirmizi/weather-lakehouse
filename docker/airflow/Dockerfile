FROM apache/airflow:2.10.1

# Baked in rather than injected through _PIP_ADDITIONAL_REQUIREMENTS: that variable
# reinstalls on every container start, needs the network to come up before the
# scheduler does, and pins nothing.
RUN pip install --no-cache-dir apache-airflow-providers-docker==3.13.0
