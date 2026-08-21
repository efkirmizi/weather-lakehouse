from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

from alerting import DEFAULT_ARGS
from credentials import s3_env

# The only job in this stack that permanently deletes data files.
#
# Iceberg's own expire_snapshots / remove_orphan_files cannot do this here: Nessie
# sets gc.enabled=false on the tables it manages, because in a Git-like catalog one
# data file may be referenced from several branches. nessie-gc is the tool that
# computes the live set across every reference before sweeping.
#
# Shipped PAUSED, unlike every other DAG here. Unpause it deliberately.
DEFAULT_CUTOFF = "NONE"

with DAG(
    dag_id="06_nessie_gc",
    description="Sweeps warehouse files that no Nessie reference points at (deletes data).",
    # Sundays at noon. Deliberately clear of the 02:00 training window (4h timeout)
    # and the 03:00 compaction: this is the only job that physically deletes files,
    # and a training run still reading an older snapshot would lose them mid-read.
    schedule="0 12 * * 0",
    start_date=datetime(2026, 8, 14),
    catchup=False,
    tags=["data-engineering", "nessie", "maintenance", "destructive"],
    default_args=DEFAULT_ARGS,
    is_paused_upon_creation=True,
    params={"gc_cutoff": DEFAULT_CUTOFF},
) as dag:

    run_gc = DockerOperator(
        task_id="run_nessie_gc",
        image="dag-nessie-gc:1.0",
        api_version="auto",
        auto_remove="success",
        network_mode="lakehouse-net",
        environment={
            **s3_env(),
            # NONE keeps every snapshot and removes only files no commit references,
            # which is the safe default. Pass a duration such as PT720H (30 days) to
            # actually expire old snapshots and reclaim their storage.
            "GC_CUTOFF": "{{ dag_run.conf.get('GC_CUTOFF', params.gc_cutoff) }}",
            # Optional ISO instant; nothing written after it can be deleted.
            "GC_MAX_FILE_MODIFICATION": "{{ dag_run.conf.get('GC_MAX_FILE_MODIFICATION', '') }}",
        },
        docker_url="unix://var/run/docker.sock",
        retries=0,  # a partially swept warehouse should be looked at, not retried blindly
        execution_timeout=timedelta(hours=2),
    )
