"""Fixtures for the suite that talks to the running stack.

Everything here needs Nessie, MinIO, MLflow and the serving API to be up, so this
suite is deliberately not part of `./dev.sh test` - that stays hermetic and instant.
Run `./dev.sh up` first, then `./dev.sh test-integration`.

This suite is the answer to the "no integration tests" line in the README's known
limitations. Until now everything touching Iceberg, MLflow or the serving path was
verified only by running a DAG and reading its log, which means it was verified
whenever someone remembered to look.

Nothing here writes to a `weather.*` table. An integration suite that can corrupt the
warehouse it is checking is not worth having, so the one test that needs to write
creates its own table in its own namespace and drops it again.
"""
import os
import uuid

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, NestedField, TimestamptzType

from lakehouse import load_iceberg_catalog

SCRATCH_NAMESPACE = "integration_scratch"


@pytest.fixture(scope="session")
def catalog():
    """The real Nessie catalog.

    Fails rather than skips when the stack is unreachable. This suite only ever runs
    because someone asked for it, and a skipped integration test is the failure mode
    tests/static/test_suite_registration.py exists to prevent: it reports nothing
    while looking green.
    """
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        assert os.environ.get(var), (
            f"{var} is unset. This suite runs against the live stack - "
            "use ./dev.sh test-integration, which passes the MinIO app credentials in."
        )
    return load_iceberg_catalog()


@pytest.fixture
def scratch_table(catalog):
    """An Iceberg table of this suite's own, dropped again afterwards."""
    if SCRATCH_NAMESPACE not in {ns[0] for ns in catalog.list_namespaces()}:
        catalog.create_namespace(SCRATCH_NAMESPACE)

    identifier = (SCRATCH_NAMESPACE, f"t_{uuid.uuid4().hex[:12]}")
    table = catalog.create_table(
        identifier,
        schema=Schema(
            NestedField(1, "timestamp", TimestamptzType(), required=True),
            NestedField(2, "value", DoubleType(), required=True),
        ),
    )
    try:
        yield table
    finally:
        catalog.drop_table(identifier)
