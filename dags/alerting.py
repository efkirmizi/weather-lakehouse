"""Failure notification shared by every Airflow DAG in this project.

Nothing here used to notice a failure: a DAG could fail every night for a week and the
only trace was a red square in a UI nobody had open. This keeps the dependency-free
default (a structured ERROR log) and adds a webhook when one is configured.
"""
import logging
import os

import requests

logger = logging.getLogger("Alerting")

# Set ALERT_WEBHOOK_URL in the scheduler environment to get a POST per failure
# (Slack incoming webhooks and anything else accepting {"text": ...} both work).
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT_SECONDS = 10


def _describe(context) -> str:
    ti = context.get("task_instance")
    exception = context.get("exception")
    parts = [
        f"dag={context.get('dag').dag_id if context.get('dag') else '?'}",
        f"task={ti.task_id if ti else '?'}",
        f"run={context.get('run_id', '?')}",
        f"try={ti.try_number if ti else '?'}",
    ]
    if exception:
        parts.append(f"error={type(exception).__name__}: {exception}")
    return " ".join(parts)


def notify_failure(context) -> None:
    """Airflow on_failure_callback. Must never raise: it runs inside the failure path."""
    message = f"Airflow task failed - {_describe(context)}"
    logger.error(message)

    if not WEBHOOK_URL:
        return

    try:
        requests.post(WEBHOOK_URL, json={"text": message}, timeout=WEBHOOK_TIMEOUT_SECONDS)
    except Exception as e:
        # A broken alerting channel must not turn one failed task into two.
        logger.error(f"Could not deliver failure notification: {e}")


# Attach to every DAG so a new one does not silently opt out of alerting.
DEFAULT_ARGS = {
    "on_failure_callback": notify_failure,
}
