"""Credential plumbing for the DockerOperator jobs.

The S3 keys used to be read with os.getenv() in seven DAG files, which meant they had
to live in the scheduler's own environment. Here they come from an Airflow connection
instead, rendered per task run: DockerOperator.environment is a template field, so
nothing touches the metadata DB at parse time.
"""
S3_CONNECTION_ID = "minio_s3"


def s3_env() -> dict:
    """Env vars every job needs to reach MinIO through the Iceberg/S3 clients."""
    return {
        "AWS_ACCESS_KEY_ID": f"{{{{ conn.{S3_CONNECTION_ID}.login }}}}",
        "AWS_SECRET_ACCESS_KEY": f"{{{{ conn.{S3_CONNECTION_ID}.password }}}}",
    }
