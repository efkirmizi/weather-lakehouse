"""Iceberg through Nessie onto MinIO, end to end.

The unit suites stub every one of these boundaries. What they cannot tell you is
whether the catalog is reachable, whether the S3 credentials in the DAG connection
actually work, or whether Iceberg returns rows in the order the whole pipeline
assumes it does not.
"""
import datetime

import pyarrow as pa

from lakehouse import scan_ordered

UTC = datetime.timezone.utc


def _rows(hours):
    """Arrow rows for the given hour offsets, in the order given."""
    base = datetime.datetime(2026, 1, 1, tzinfo=UTC)
    return pa.Table.from_pydict(
        {
            "timestamp": [base + datetime.timedelta(hours=h) for h in hours],
            "value": [float(h) for h in hours],
        },
        # nullable=False on purpose: the Iceberg schema declares both fields
        # required, and pyiceberg refuses an append whose Arrow schema is laxer.
        schema=pa.schema([
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
        ]),
    )


def test_a_table_survives_a_write_and_a_read(scratch_table):
    """Catalog, credentials and object store in one assertion. If MinIO's bucket
    policy or the connection's keys are wrong, this is where it shows - not three
    quarters of the way through a training run."""
    scratch_table.append(_rows([0, 1, 2]))

    read_back = scratch_table.scan().to_arrow()

    assert read_back.num_rows == 3
    assert sorted(read_back.column("value").to_pylist()) == [0.0, 1.0, 2.0]


def test_scan_ordered_sorts_rows_iceberg_hands_back_unordered(scratch_table):
    """The convention the entire pipeline rests on.

    Iceberg guarantees no ordering across data files, and every consumer here slices
    sequences positionally - build_context_window takes the last SEQ_LEN rows and
    trusts they are the last SEQ_LEN hours. scan_ordered is what makes that true.
    Appending in three separate commits, deliberately out of order, is the closest
    this suite can get to reproducing what 05_lakehouse_maintenance's
    rewrite_data_files can do to file order at any time.
    """
    for hour in (5, 1, 3):
        scratch_table.append(_rows([hour]))

    ordered = scan_ordered(scratch_table, ("timestamp", "value"))

    assert ordered.column("value").to_pylist() == [1.0, 3.0, 5.0]


def test_the_warehouse_holds_the_tables_the_pipeline_expects(catalog):
    """A missing table here means a DAG has not run, and every downstream number is
    stale rather than wrong - which is harder to notice."""
    tables = {name for _, name in catalog.list_tables("weather")}

    assert {"observations", "ml_features", "scaling_parameters"} <= tables
