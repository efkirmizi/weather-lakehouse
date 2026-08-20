"""Guards Airflow's DAG_DISCOVERY_SAFE_MODE requirement.

Airflow only parses a file under dags/ if it contains BOTH the strings "airflow" and
"dag". A file that defines a DAG purely through a helper import satisfies neither, and
Airflow then skips it *without reporting an import error* - the DAG simply vanishes
from the UI. That happened here when 03a/03b were reduced to factory calls.

Pure stdlib on purpose: this has to run before anything is built.
"""
import pathlib

DAGS_DIR = pathlib.Path(__file__).resolve().parents[2] / "dags"


def _ignored_files():
    ignore = DAGS_DIR / ".airflowignore"
    if not ignore.exists():
        return set()
    return {
        line.strip()
        for line in ignore.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_every_dag_file_is_discoverable():
    ignored = _ignored_files()
    offenders = []

    for path in sorted(DAGS_DIR.glob("*.py")):
        if path.name in ignored:
            continue
        text = path.read_text().lower()
        missing = [token for token in ("airflow", "dag") if token not in text]
        if missing:
            offenders.append(f"{path.name} is missing {missing}")

    assert not offenders, (
        "Airflow will silently skip these files (DAG_DISCOVERY_SAFE_MODE needs both "
        "'airflow' and 'dag' in the source):\n  " + "\n  ".join(offenders)
    )


def test_airflowignore_entries_exist():
    """A stale ignore entry means a helper is being parsed as a DAG file again."""
    missing = [name for name in _ignored_files() if not (DAGS_DIR / name).exists()]
    assert not missing, f".airflowignore lists files that no longer exist: {missing}"
