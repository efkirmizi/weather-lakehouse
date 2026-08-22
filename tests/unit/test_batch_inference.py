"""Row construction and table creation for the forecast table. Runs inside
dag-pytorch-model-training."""
import datetime

from batch_inference import forecast_rows, open_predictions_table
from lakehouse import OUTPUT_CHANNELS, PRED_LEN

NOW = datetime.datetime(2026, 8, 22, 0, 0, tzinfo=datetime.timezone.utc)


def _timestamps():
    return [NOW + datetime.timedelta(hours=i) for i in range(1, PRED_LEN + 1)]


def _predictions():
    return [[float(hour)] * OUTPUT_CHANNELS for hour in range(1, PRED_LEN + 1)]


def test_every_predicted_hour_carries_its_horizon():
    """Residuals averaged over all 24 horizons hide that error more than doubles
    across them. The horizon is known here and nowhere downstream."""
    rows = forecast_rows("M", "3", _timestamps(), _predictions(), NOW)
    assert rows["horizon"] == list(range(1, PRED_LEN + 1))


def test_the_horizon_lines_up_with_the_timestamp_it_describes():
    rows = forecast_rows("M", "3", _timestamps(), _predictions(), NOW)
    for horizon, timestamp in zip(rows["horizon"], rows["forecast_timestamp"]):
        assert timestamp == NOW + datetime.timedelta(hours=horizon)


def test_every_column_has_one_entry_per_predicted_hour():
    rows = forecast_rows("M", "3", _timestamps(), _predictions(), NOW)
    assert {len(v) for v in rows.values()} == {PRED_LEN}


def test_the_model_identity_is_repeated_on_every_row():
    rows = forecast_rows("M", "3", _timestamps(), _predictions(), NOW)
    assert set(rows["model_name"]) == {"M"}
    assert set(rows["model_version"]) == {"3"}


class _FakeCatalog:
    """load_table fails exactly as it would against a catalog that has never seen
    this table; create_table records the schema instead of touching a real one, so
    the field ids the migration would actually commit to can be asserted without a
    running Nessie/Iceberg stack."""

    def __init__(self):
        self.created_schema = None

    def load_table(self, _identifier):
        raise Exception("weather.forecast_predictions does not exist")

    def create_table(self, _identifier, schema, partition_spec):
        self.created_schema = schema
        return "created-table"


def test_open_predictions_table_creates_horizon_at_a_stable_field_id():
    """forecast_predictions is created exactly once, at the migration, with
    horizon required=True - a field-id slip here is not fixable in place the way a
    bug in forecast_rows above is; the only recovery is dropping the table and
    re-forecasting. Pins the ids the migration would actually commit to before
    that happens, and that fields 1, 2 (and its list element, 3), 4, 5 and 6 are
    undisturbed by adding horizon."""
    catalog = _FakeCatalog()

    open_predictions_table(catalog)

    fields = {f.field_id: f for f in catalog.created_schema.fields}
    assert fields[7].name == "horizon"
    assert fields[7].required is True

    assert fields[1].name == "forecast_timestamp"
    assert fields[2].name == "predicted_features"
    assert fields[2].field_type.element_id == 3
    assert fields[4].name == "model_name"
    assert fields[5].name == "model_version"
    assert fields[6].name == "created_at"
