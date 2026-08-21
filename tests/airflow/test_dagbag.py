"""Parses the DAG folder exactly the way the scheduler does.

Runs inside the Airflow image with /opt/airflow/dags mounted; needs no metadata DB.
DagBag applies DAG_DISCOVERY_SAFE_MODE, so a file the scheduler would silently skip
shows up here as a missing DAG rather than as an import error.
"""
import pathlib
import warnings

from airflow.models import DagBag

DAGS_DIR = pathlib.Path("/opt/airflow/dags")


def _expected_dag_count():
    """One DAG per non-ignored .py file. Helper modules live in .airflowignore."""
    ignore = DAGS_DIR / ".airflowignore"
    ignored = set()
    if ignore.exists():
        ignored = {
            line.strip() for line in ignore.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    return sum(1 for p in DAGS_DIR.glob("*.py") if p.name not in ignored)


def test_dagbag_has_no_import_errors():
    bag = DagBag(str(DAGS_DIR), include_examples=False)
    assert not bag.import_errors, f"DAG import errors: {bag.import_errors}"


def test_every_dag_file_produces_a_dag():
    bag = DagBag(str(DAGS_DIR), include_examples=False)
    expected = _expected_dag_count()
    assert len(bag.dags) == expected, (
        f"expected {expected} DAGs (one per non-ignored file), found {len(bag.dags)}: "
        f"{sorted(bag.dags)}. A file Airflow skipped reports no import error."
    )


def test_every_dag_alerts_on_failure():
    bag = DagBag(str(DAGS_DIR), include_examples=False)
    missing = [
        dag_id for dag_id, dag in bag.dags.items()
        if not dag.default_args.get("on_failure_callback")
    ]
    assert not missing, f"DAGs with no failure notification: {missing}"


def test_gpu_tasks_are_serialized_by_the_pool():
    """One physical GPU, and every GPU task asks for all of them."""
    bag = DagBag(str(DAGS_DIR), include_examples=False)
    offenders = []
    for dag_id, dag in bag.dags.items():
        for task in dag.tasks:
            if getattr(task, "device_requests", None) and task.pool != "single_gpu":
                offenders.append(f"{dag_id}.{task.task_id}")
    assert not offenders, f"GPU tasks outside the single_gpu pool: {offenders}"


def test_no_dag_file_uses_a_deprecated_argument():
    """A deprecated argument still parses, so nothing fails - it only warns.

    `auto_remove=True` is the live example: providers-docker 3.13.0 converts the bool
    to 'success' and warns, and a later release drops the conversion and raises
    ValueError at parse time instead. Asserting on `task.auto_remove` cannot catch
    that, because by the time the operator exists the value has already been
    converted - the warning is the only signal, so this is what has to be asserted on.

    Scoped to warnings raised *from* dags/. Airflow 2.10.1's own dependency stack
    (Flask, marshmallow, SQLAlchemy) emits four more that we cannot act on, and
    failing this suite on those would only teach everyone to ignore it.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DagBag(str(DAGS_DIR), include_examples=False)

    offenders = sorted({
        f"{pathlib.Path(w.filename).name}:{w.lineno} - {w.category.__name__}: {w.message}"
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and pathlib.Path(w.filename).parent == DAGS_DIR
    })
    assert not offenders, (
        "DAG files use deprecated arguments:\n  " + "\n  ".join(offenders)
    )
