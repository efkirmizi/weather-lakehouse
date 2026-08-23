"""The contract three images agree on and none of them can import.

`weather.scaling_parameters` is written by the feature engineering image and read by
the training image and the serving image. Those three cannot import each other, so
the agreement lives entirely in a published table - and the only thing that can check
it is a test that reads the real one. A unit test can stub the table, but stubbing it
means asserting that this file's idea of the layout matches this file's idea of the
layout.

The failure this guards against is silent. Nothing raises if the channels shift: the
serving API de-normalizes channel 0 with whatever parameters sit at index 0 and draws
the result on a chart labelled "temperature".
"""
from lakehouse import INPUT_CHANNELS, OUTPUT_CHANNELS, TEMPERATURE_CHANNEL, scan_ordered

# The names the serving layer and the drift monitor expect at the front of the vector,
# in order. Deliberately spelled out rather than imported: importing them from the
# writer would make this test agree with itself.
FORECAST_CHANNEL_NAMES = ["temperature", "humidity", "precipitation", "wind_speed"]


def _scaling_rows(catalog):
    table = catalog.load_table(("weather", "scaling_parameters"))
    arrow = table.scan().to_arrow().to_pylist()
    return sorted(arrow, key=lambda r: r["feature_index"])


def _cyclical_indices(rows):
    """Channels published as sin/cos pairs, found by name rather than by a hardcoded
    offset. The boundary is len(SCALED_COLUMNS) in the feature engineering image,
    which this one cannot import - and hardcoding 10 here would keep passing, on the
    wrong channels, the day an eleventh standardized variable is added."""
    return [r["feature_index"] for r in rows if r["feature_name"].endswith(("_sin", "_cos"))]


def test_the_scaling_table_covers_every_channel_exactly_once(catalog):
    rows = _scaling_rows(catalog)

    assert len(rows) == INPUT_CHANNELS
    assert [r["feature_index"] for r in rows] == list(range(INPUT_CHANNELS))
    assert len({r["feature_name"] for r in rows}) == INPUT_CHANNELS


def test_the_forecast_channels_sit_where_every_consumer_reads_them(catalog):
    """batch_inference writes channel TEMPERATURE_CHANNEL as the forecast, the API
    de-normalizes it with the row at that index, and the promotion gate reports its
    RMSE in degrees. All three take the position on faith."""
    rows = _scaling_rows(catalog)

    assert [r["feature_name"] for r in rows[:OUTPUT_CHANNELS]] == FORECAST_CHANNEL_NAMES
    assert rows[TEMPERATURE_CHANNEL]["feature_name"] == "temperature"


def test_temperature_is_published_with_a_usable_inverse_transform(catalog):
    """std == 0 would make the API's inverse transform a no-op and every forecast
    read as a fraction of a degree; a negative one would invert the sign."""
    temperature = _scaling_rows(catalog)[TEMPERATURE_CHANNEL]

    assert temperature["std_value"] > 0
    assert -50.0 < temperature["mean_value"] < 50.0


def test_the_cyclical_tail_is_published_as_identity(catalog):
    """Channels past the standardized block are already bounded in [-1, 1]. They
    still get a row so no consumer has to know which channels are silently absent,
    and that row has to be the identity transform or the API will 'de-normalize'
    a sine wave."""
    rows = _scaling_rows(catalog)
    cyclical = set(_cyclical_indices(rows))
    assert cyclical, "no sin/cos channels published at all"

    for row in rows:
        if row["feature_index"] in cyclical:
            assert (row["mean_value"], row["std_value"]) == (0.0, 1.0), row["feature_name"]


def test_the_feature_vector_is_as_wide_as_the_contract_says(catalog):
    """INPUT_CHANNELS is a constant in the training image. ml_features is written by
    a different image. A model built for 16 and fed 4 fails loudly; built for 16 and
    fed 16 in a different order, it does not."""
    table = catalog.load_table(("weather", "ml_features"))
    latest = scan_ordered(table, ("timestamp", "features")).column("features").to_pylist()[-1]

    assert len(latest) == INPUT_CHANNELS


def test_the_published_cyclical_channels_stay_on_the_unit_circle(catalog):
    """sin^2 + cos^2 == 1 for each pair. Catches a period applied to the wrong
    channel, or a calendar part read from a container in the wrong timezone, both of
    which leave the values in range and the vector the right width."""
    table = catalog.load_table(("weather", "ml_features"))
    latest = scan_ordered(table, ("timestamp", "features")).column("features").to_pylist()[-1]

    cyclical = _cyclical_indices(_scaling_rows(catalog))
    assert len(cyclical) % 2 == 0, f"sin/cos channels do not pair up: {cyclical}"

    for sin_index, cos_index in zip(cyclical[::2], cyclical[1::2]):
        sin, cos = latest[sin_index], latest[cos_index]
        assert abs(sin * sin + cos * cos - 1.0) < 1e-5, f"channels {sin_index},{cos_index}"
