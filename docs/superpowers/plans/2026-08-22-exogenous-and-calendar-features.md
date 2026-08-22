# Exogenous and Calendar Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take `ml_features` from 4 to 16 channels — adding six exogenous Open-Meteo variables and four calendar encodings — and make per-horizon forecast error visible everywhere it is judged.

**Architecture:** The channel layout is declared exactly once per image and travels between images through `weather.scaling_parameters`. The original four channels keep positions 0-3, so `denormalize(x[0])` in the serving layer never changes. Model output stays 4 channels; input becomes 16.

**Tech Stack:** PySpark 3.5.2, PyIceberg 0.11.1, PyTorch 2.7.1, MLflow 3.15.1, FastAPI, Streamlit, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-exogenous-and-calendar-features-design.md`

## Global Constraints

- Channel order is fixed by the spec's table. Channels 0-3 are `temperature, humidity, precipitation, wind_speed` and must not move.
- Channels 10-15 (`wind_dir_sin/cos`, `hour_sin/cos`, `doy_sin/cos`) are **never** standardized; they are published to `scaling_parameters` with `mean=0.0, std=1.0`.
- Model `output_dim` stays 4. `input_dim` becomes 16.
- Architecture and hyperparameters do **not** change: `d_model=32`, `n_heads=2`, `num_layers=4`, `dim_feedforward=128`, `hidden_dim=64`, `dropout` unchanged, `batch_size=128`, `patience=4`, `weight_decay=1e-4`.
- `seq_len=72`, `pred_len=24` everywhere.
- Comments explain *why*, citing the failure that motivated the code. A comment restating what a line does is out of place in this codebase.
- Every new `tests/unit/*.py` file must be registered in **both** `dev.sh` and `.github/workflows/ci.yml`; `tests/static/test_suite_registration.py` fails the build otherwise.
- Run the relevant suite in its real image, per `dev.sh`. Never claim a pass without the output.

---

### Task 1: One shared forecast contract

`SEQ_LEN`/`PRED_LEN` are currently written as literals in `batch_inference.py`, `evaluate_and_promote.py` and both training configs. They must agree or the pipeline silently slices the wrong windows. Add the channel widths beside them.

**Files:**
- Modify: `jobs/model_training/lakehouse.py`
- Modify: `jobs/model_training/batch_inference.py:26-28`
- Modify: `jobs/model_training/evaluate_and_promote.py:22-23`
- Modify: `jobs/model_training/train_lstm.py:21-27`
- Modify: `jobs/model_training/train_transformer.py:21-28`
- Create: `tests/unit/test_contract.py`
- Modify: `dev.sh`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: `lakehouse.SEQ_LEN = 72`, `lakehouse.PRED_LEN = 24`, `lakehouse.INPUT_CHANNELS = 16`, `lakehouse.OUTPUT_CHANNELS = 4`, `lakehouse.TEMPERATURE_CHANNEL = 0`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_contract.py`:

```python
"""The window and channel geometry every job in this image has to agree on.

Runs inside dag-pytorch-model-training (PYTHONPATH=/app).
"""
import batch_inference
import evaluate_and_promote
import lakehouse


def test_the_contract_declares_the_geometry():
    assert lakehouse.SEQ_LEN == 72
    assert lakehouse.PRED_LEN == 24
    assert lakehouse.INPUT_CHANNELS == 16
    assert lakehouse.OUTPUT_CHANNELS == 4
    assert lakehouse.TEMPERATURE_CHANNEL == 0


def test_only_the_output_channels_are_forecast():
    """Calendar channels are deterministic and the exogenous ones are inputs;
    forecasting them would make the task harder for nothing."""
    assert lakehouse.OUTPUT_CHANNELS < lakehouse.INPUT_CHANNELS


def test_every_consumer_reads_the_same_window_geometry():
    """These used to be literals in three files. A mismatch does not raise - it
    slices a different window and reports a confident wrong number."""
    assert batch_inference.SEQ_LEN == lakehouse.SEQ_LEN
    assert batch_inference.PRED_LEN == lakehouse.PRED_LEN
    assert evaluate_and_promote.SEQ_LEN == lakehouse.SEQ_LEN
    assert evaluate_and_promote.PRED_LEN == lakehouse.PRED_LEN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm --user root -v "$PWD/tests":/tests -e PYTHONPATH=/app dag-pytorch-model-training:1.0 sh -c "pip install -q pytest && python -m pytest /tests/unit/test_contract.py -q -p no:cacheprovider"`

Expected: FAIL — `AttributeError: module 'lakehouse' has no attribute 'SEQ_LEN'`.

- [ ] **Step 3: Add the constants to `lakehouse.py`**

Insert after the `ONNX_OPSET = 17` line:

```python
# The window and channel geometry every job in this image has to agree on. These were
# literals in three separate files; a mismatch does not raise anything, it just slices
# a different window and reports a confident wrong number.
SEQ_LEN = 72
PRED_LEN = 24

# ml_features carries 16 channels (see 02's channel declaration and the published
# scaling_parameters); only the first four - temperature, humidity, precipitation,
# wind_speed - are forecast. The rest are exogenous or deterministic inputs.
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 4
# Position of temperature, which is what the serving layer de-normalizes and what the
# per-horizon benchmark reports.
TEMPERATURE_CHANNEL = 0
```

- [ ] **Step 4: Point the consumers at it**

In `jobs/model_training/batch_inference.py`, replace the `SEQ_LEN = 72` / `PRED_LEN = 24` lines with an import. The existing import line becomes:

```python
from lakehouse import (
    load_iceberg_catalog, scan_ordered, ONNX_ARTIFACT_NAME, SEQ_LEN, PRED_LEN,
)
```

and delete the two literal assignments beneath `MODEL_PREFIX`.

In `jobs/model_training/evaluate_and_promote.py`, delete the `SEQ_LEN = 72` / `PRED_LEN = 24` lines and add to the imports:

```python
from lakehouse import OUTPUT_CHANNELS, PRED_LEN, SEQ_LEN, TEMPERATURE_CHANNEL
```

In `jobs/model_training/train_lstm.py`, add `from lakehouse import INPUT_CHANNELS, OUTPUT_CHANNELS, PRED_LEN, SEQ_LEN` to the imports and change CONFIG:

```python
CONFIG = {
    "model_registry_name": "Weather_Forecaster_FastLSTM",
    "table_name": "weather.ml_features",
    "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "batch_size": 128,
    "input_dim": INPUT_CHANNELS, "output_dim": OUTPUT_CHANNELS,
    "hidden_dim": 64, "dropout": 0.2,
    "patience": 4, "weight_decay": 1e-4
}
```

In `jobs/model_training/train_transformer.py`, the same import and:

```python
CONFIG = {
    "model_registry_name": "Weather_Forecaster_Transformer",
    "table_name": "weather.ml_features",
    "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "batch_size": 128,
    "input_dim": INPUT_CHANNELS, "output_dim": OUTPUT_CHANNELS,
    "d_model": 32, "n_heads": 2,
    "num_layers": 4, "dim_feedforward": 128, "dropout": 0.1,
    "patience": 4, "weight_decay": 1e-4
}
```

Both files still reference `CONFIG["feature_dim"]` at their model construction call; leave those broken for now — Task 5 fixes them and has the test that proves it.

- [ ] **Step 5: Register the new test file in both runners**

In `dev.sh`, append `/tests/unit/test_contract.py` to the training image's `run_pytest` argument list.
In `.github/workflows/ci.yml`, add `tests/unit/test_contract.py \` to the training job's pytest arguments.

- [ ] **Step 6: Run the tests**

Run: `python3 -m pytest tests/static -q` — expected PASS (5 tests; the registration guard is satisfied).
Run the training-image command from Step 2 — expected PASS.

- [ ] **Step 7: Commit**

```bash
git add jobs/model_training/lakehouse.py jobs/model_training/batch_inference.py \
        jobs/model_training/evaluate_and_promote.py jobs/model_training/train_lstm.py \
        jobs/model_training/train_transformer.py tests/unit/test_contract.py \
        dev.sh .github/workflows/ci.yml
git commit -m "Declare the window and channel geometry in one place"
```

---

### Task 2: ETL fetches the exogenous variables

The SDK returns variables positionally — `hourly.Variables(i)` — in request order, and the current code hand-indexes four of them. At eleven variables that is a live hazard: reordering the request list silently swaps columns and nothing catches it.

**Files:**
- Modify: `jobs/weather_etl/weather_etl.py:33-38, 136-158`
- Modify: `tests/unit/test_weather_etl.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `weather_etl.HOURLY_VARIABLES` — a list of `(api_name, column, (low, high))` tuples; `weather_etl.DEW_POINT_TOLERANCE_C = 0.5`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_weather_etl.py`:

```python
import pandas as pd
import pytest

from weather_etl import (
    DEFAULT_PARAMS,
    HOURLY_VARIABLES,
    run_data_quality_checks,
)


def _valid_frame(**overrides):
    """One physically plausible hour, with every column the ETL now ingests."""
    row = {
        "timestamp": pd.Timestamp("2026-08-20 12:00", tz="UTC"),
        "temperature_c": 25.0,
        "humidity_percent": 55.0,
        "precipitation_mm": 0.0,
        "wind_speed_kmh": 12.0,
        "pressure_msl_hpa": 1013.0,
        "dew_point_c": 15.0,
        "cloud_cover_percent": 40.0,
        "shortwave_radiation_wm2": 700.0,
        "soil_temperature_c": 27.0,
        "soil_moisture_m3m3": 0.21,
        "wind_direction_deg": 210.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_the_request_list_is_generated_from_the_variable_table():
    """The SDK returns variables positionally, so the request order and the column
    order must come from one declaration or they drift apart silently."""
    assert DEFAULT_PARAMS["hourly"] == [api for api, _, _ in HOURLY_VARIABLES]


def test_a_valid_hour_passes_every_gate():
    run_data_quality_checks(_valid_frame())


@pytest.mark.parametrize("column,bad", [
    ("pressure_msl_hpa", 1200.0),
    ("cloud_cover_percent", 140.0),
    ("shortwave_radiation_wm2", -5.0),
    ("soil_moisture_m3m3", 1.4),
    ("wind_direction_deg", 400.0),
])
def test_each_new_column_has_a_physical_bound(column, bad):
    with pytest.raises(ValueError, match=column):
        run_data_quality_checks(_valid_frame(**{column: bad}))


def test_dew_point_above_temperature_is_rejected():
    """Air cannot hold more moisture than saturation. A violation means the two
    columns have been misaligned - which no per-column bound check can see, because
    both values are individually plausible."""
    with pytest.raises(ValueError, match="dew point"):
        run_data_quality_checks(_valid_frame(temperature_c=10.0, dew_point_c=18.0))


def test_saturation_noise_is_tolerated():
    """At saturation ERA5 puts the dew point a hair above the temperature."""
    run_data_quality_checks(_valid_frame(temperature_c=10.0, dew_point_c=10.3))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker run --rm --user root -v "$PWD/tests":/tests -e PYTHONPATH=/opt/spark/work-dir dag-pyspark-etl:1.0 sh -c "pip install -q pytest && python -m pytest /tests/unit/test_weather_etl.py -q -p no:cacheprovider"`

Expected: FAIL — `ImportError: cannot import name 'HOURLY_VARIABLES'`.

- [ ] **Step 3: Declare the variables**

In `jobs/weather_etl/weather_etl.py`, replace the `DEFAULT_PARAMS` block with:

```python
# One ordered declaration of what we pull, what we call it, and what counts as
# physically possible. The Open-Meteo SDK hands variables back positionally -
# hourly.Variables(i), in request order - so the request list and the column list
# have to be generated from the same source. Hand-indexing four was survivable;
# hand-indexing eleven means a reordering silently swaps two columns and every
# downstream number stays plausible.
HOURLY_VARIABLES = [
    # (Open-Meteo name,          our column,                (low, high))
    ("temperature_2m",            "temperature_c",           (-60.0, 60.0)),
    ("relative_humidity_2m",      "humidity_percent",        (0.0, 100.0)),
    ("precipitation",             "precipitation_mm",        (0.0, 500.0)),
    ("wind_speed_10m",            "wind_speed_kmh",          (0.0, 300.0)),
    ("pressure_msl",              "pressure_msl_hpa",        (870.0, 1085.0)),
    ("dew_point_2m",              "dew_point_c",             (-60.0, 60.0)),
    ("cloud_cover",               "cloud_cover_percent",     (0.0, 100.0)),
    ("shortwave_radiation",       "shortwave_radiation_wm2", (0.0, 1400.0)),
    ("soil_temperature_0_to_7cm", "soil_temperature_c",      (-60.0, 70.0)),
    ("soil_moisture_0_to_7cm",    "soil_moisture_m3m3",      (0.0, 1.0)),
    ("wind_direction_10m",        "wind_direction_deg",      (0.0, 360.0)),
]

# ERA5 puts the dew point a hair above the temperature at saturation.
DEW_POINT_TOLERANCE_C = 0.5

DEFAULT_PARAMS = {
    "latitude": 40.98,
    "longitude": 27.51,
    "hourly": [api_name for api_name, _, _ in HOURLY_VARIABLES],
}
```

- [ ] **Step 4: Generate the frame from the declaration**

Replace the body of `transform_weather_data`:

```python
def transform_weather_data(response: Any) -> pd.DataFrame:
    """Transforms raw API response into a structured pandas DataFrame."""
    hourly = response.Hourly()
    hourly_data = {
        "timestamp": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
    }
    # Positional, in the order they were requested - which is why both come from
    # HOURLY_VARIABLES rather than being written out twice.
    for index, (_, column, _) in enumerate(HOURLY_VARIABLES):
        hourly_data[column] = hourly.Variables(index).ValuesAsNumpy()

    df = pd.DataFrame(data=hourly_data)
    # Drop any null rows just in case the API returned gaps for very old dates.
    # The first 7 hours of 1940-01-01 are the known case: precipitation,
    # shortwave_radiation and wind_gusts need a preceding accumulation interval.
    df.dropna(inplace=True)
    return df
```

- [ ] **Step 5: Drive the quality gate from the declaration**

Replace the bounds section of `run_data_quality_checks` (keep the null check and the
sequential-continuity warning exactly as they are):

```python
    # 2. Physical Range Bounds, from the same table that defines the columns.
    for _, column, (low, high) in HOURLY_VARIABLES:
        if not df[column].between(low, high).all():
            raise ValueError(
                f"Data Quality Failure: {column} outside its physical range "
                f"[{low}, {high}]."
            )

    # 2b. A cross-column invariant, which is the only kind that catches a column
    # mix-up: both values stay individually plausible when two columns are swapped.
    # Air cannot hold more moisture than saturation, so the dew point can never
    # exceed the temperature.
    excess = df["dew_point_c"] - df["temperature_c"]
    if (excess > DEW_POINT_TOLERANCE_C).any():
        raise ValueError(
            f"Data Quality Failure: dew point exceeds temperature by up to "
            f"{excess.max():.2f} C. The columns are probably misaligned."
        )
```

- [ ] **Step 6: Run the tests**

Run the Step 2 command — expected PASS.

- [ ] **Step 7: Commit**

```bash
git add jobs/weather_etl/weather_etl.py tests/unit/test_weather_etl.py
git commit -m "Ingest the exogenous variables a 24-hour forecast depends on"
```

---

### Task 3: The 16-channel feature vector

**Files:**
- Modify: `jobs/feature_engineering/feature_engineering.py:19-22, 84-90, 116-145`
- Modify: `tests/unit/test_feature_engineering.py`

**Interfaces:**
- Consumes: the ETL columns from Task 2.
- Produces: `feature_engineering.SCALED_COLUMNS` (10 `(column, name)` pairs), `feature_engineering.CYCLICAL_CHANNELS` (3 `(name, period, value_fn)` tuples), `feature_engineering.FEATURE_COLS` (10 column names), `feature_engineering.SERVING_FEATURE_NAMES` (16 names in channel order).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_feature_engineering.py`:

```python
import math

from feature_engineering import (
    CYCLICAL_CHANNELS,
    FEATURE_COLS,
    SCALED_COLUMNS,
    SERVING_FEATURE_NAMES,
    cyclical_pair,
)


def test_the_first_four_channels_never_move():
    """The serving API de-normalizes channel 0 as temperature from an image that
    cannot import this layout at all, and drift_monitor/evaluate_and_promote read it
    through TEMPERATURE_CHANNEL, which only agrees with this file by convention.
    Moving these silently re-labels every forecast the dashboard draws."""
    assert SERVING_FEATURE_NAMES[:4] == [
        "temperature", "humidity", "precipitation", "wind_speed"
    ]


def test_the_vector_is_sixteen_channels_and_names_are_unique():
    assert len(SERVING_FEATURE_NAMES) == 16
    assert len(set(SERVING_FEATURE_NAMES)) == 16


def test_scaled_columns_and_feature_cols_stay_in_step():
    """FEATURE_COLS drives standardization; SERVING_FEATURE_NAMES drives the published
    table. They are two views of one list and must not drift."""
    assert FEATURE_COLS == [column for column, _ in SCALED_COLUMNS]
    assert SERVING_FEATURE_NAMES[:len(SCALED_COLUMNS)] == [n for _, n in SCALED_COLUMNS]


def test_cyclical_channels_come_last_as_sin_cos_pairs():
    tail = SERVING_FEATURE_NAMES[len(SCALED_COLUMNS):]
    expected = [f"{name}_{part}" for name, _, _ in CYCLICAL_CHANNELS
                for part in ("sin", "cos")]
    assert tail == expected


def test_a_compass_bearing_wraps_without_a_seam():
    """359 degrees and 1 degree are two degrees apart. On a raw scale they are the
    two extremes, which is why direction is encoded as sin/cos at all."""
    near_north = cyclical_pair(359.0, 360.0)
    just_past_north = cyclical_pair(1.0, 360.0)
    distance = math.dist(near_north, just_past_north)
    assert distance < 0.05


def test_new_years_eve_and_new_years_day_are_neighbours():
    end = cyclical_pair(365.0, 365.25)
    start = cyclical_pair(1.0, 365.25)
    assert math.dist(end, start) < 0.05


def test_midnight_and_twenty_three_hundred_are_neighbours():
    assert math.dist(cyclical_pair(23.0, 24.0), cyclical_pair(0.0, 24.0)) < 0.3


def test_cyclical_values_stay_inside_the_unit_circle():
    """They are published with identity scaling, so they have to already be bounded."""
    for value in (0.0, 90.0, 180.0, 270.0, 359.0):
        for component in cyclical_pair(value, 360.0):
            assert -1.0 <= component <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker run --rm --user root -v "$PWD/tests":/tests -e PYTHONPATH=/opt/spark/work-dir dag-pyspark-feature-engineering:1.0 sh -c "pip install -q pytest && python -m pytest /tests/unit/test_feature_engineering.py -q -p no:cacheprovider"`

Expected: FAIL — `ImportError: cannot import name 'SCALED_COLUMNS'`.

- [ ] **Step 3: Declare the channels**

In `jobs/feature_engineering/feature_engineering.py`, add `import math` at the top and replace the `FEATURE_COLS` / `SERVING_FEATURE_NAMES` block with:

```python
# THE feature vector, in order, declared once. Five functions in this file used to
# each carry a piece of this knowledge; with three kinds of channel that scatters
# badly, and the layout is what README calls load-bearing.
#
# Channels 0-3 keep their identity and position forever: the serving API reads
# channel 0 as temperature from an image with no copy of this layout, and
# drift_monitor/evaluate_and_promote read it through TEMPERATURE_CHANNEL, which
# only agrees with this file by convention.
SCALED_COLUMNS = [
    # (source column, channel name)
    ("temperature_c",           "temperature"),
    ("humidity_percent",        "humidity"),
    ("precipitation_mm",        "precipitation"),
    ("wind_speed_kmh",          "wind_speed"),
    ("pressure_msl_hpa",        "pressure"),
    ("dew_point_c",             "dew_point"),
    ("cloud_cover_percent",     "cloud_cover"),
    ("shortwave_radiation_wm2", "shortwave_radiation"),
    ("soil_temperature_c",      "soil_temperature"),
    ("soil_moisture_m3m3",      "soil_moisture"),
]

# Emitted as sin/cos pairs, never standardized: a compass bearing and a clock are
# circular, so on a linear scale 359 and 1 sit at opposite extremes and 23:00 looks
# maximally far from midnight. Each contributes two channels, already in [-1, 1].
CYCLICAL_CHANNELS = [
    # (name prefix, period, raw value)
    ("wind_dir", 360.0,  lambda: F.coalesce(F.col("wind_direction_deg"), F.lit(0.0))),
    ("hour",      24.0,  lambda: F.hour("timestamp")),
    ("doy",      365.25, lambda: F.dayofyear("timestamp")),
]

FEATURE_COLS = [column for column, _ in SCALED_COLUMNS]
SERVING_FEATURE_NAMES = (
    [name for _, name in SCALED_COLUMNS]
    + [f"{name}_{part}" for name, _, _ in CYCLICAL_CHANNELS for part in ("sin", "cos")]
)


def cyclical_pair(value, period):
    """(sin, cos) of a value on a circle of the given period.

    Plain Python beside the Spark version so the wrap-around is testable without a
    session; both use the same formula.
    """
    angle = 2.0 * math.pi * float(value) / period
    return math.sin(angle), math.cos(angle)
```

- [ ] **Step 4: Emit all sixteen channels**

Replace `standardize`:

```python
def standardize(df: DataFrame, stats: dict) -> DataFrame:
    """Imputes nulls with the feature mean, standardizes, and appends the cyclical
    channels, all in one projection.

    Applying the constants directly is what makes an incremental run possible: new
    rows can be normalized with exactly the parameters the existing rows already use.
    """
    columns = []
    for c in FEATURE_COLS:
        mean, std = stats[c]
        filled = F.coalesce(F.col(c), F.lit(mean))
        columns.append(((filled - F.lit(mean)) / F.lit(std)).cast("float"))

    for _, period, value_fn in CYCLICAL_CHANNELS:
        angle = (2.0 * math.pi / period) * value_fn()
        columns.append(F.sin(angle).cast("float"))
        columns.append(F.cos(angle).cast("float"))

    return df.select(F.col("timestamp"), F.array(*columns).alias("features"))
```

- [ ] **Step 5: Publish all sixteen, and fix the lookup**

Replace `write_scaling_parameters`:

```python
def write_scaling_parameters(spark: SparkSession, stats: dict) -> None:
    """Publishes the inverse transform for the serving layer.

    Only ever called on a full rebuild: on an incremental run the existing rows were
    normalized with the published values, so replacing them would silently make the
    stored vectors and the de-normalization disagree.

    The cyclical channels are published too, with identity scaling. They need no
    inverse transform, but a consumer reading this table must not have to know which
    channels it silently omits.
    """
    rows = [
        (name, index, stats[column][0], stats[column][1])
        for index, (column, name) in enumerate(SCALED_COLUMNS)
    ]
    rows += [
        (name, len(SCALED_COLUMNS) + offset, 0.0, 1.0)
        for offset, name in enumerate(SERVING_FEATURE_NAMES[len(SCALED_COLUMNS):])
    ]
    for name, _, mean_val, std_val in rows:
        logger.info(f"Scaling parameter -> {name}: mean={mean_val:.4f}, std={std_val:.4f}")

    schema = "feature_name STRING, feature_index INT, mean_value DOUBLE, std_value DOUBLE"
    spark.createDataFrame(rows, schema=schema).write \
        .format("iceberg") \
        .mode("overwrite") \
        .saveAsTable(SCALING_TABLE)
    logger.info(f"Wrote {len(rows)} scaling parameters to {SCALING_TABLE}.")
```

In `load_published_stats`, the mapping must zip against the **standardized** names only — the two lists stopped being the same length here, and zipping the full name list would pair `temperature_c` with the wrong parameters on an incremental run:

```python
    try:
        return {
            column: by_name[name] for column, name in SCALED_COLUMNS
        }
    except KeyError:
        return None
```

- [ ] **Step 6: Run the tests**

Run the Step 2 command — expected PASS (the pre-existing drift tests still pass because temperature, humidity and precipitation keep positions 0-2 in `FEATURE_COLS`).

- [ ] **Step 7: Commit**

```bash
git add jobs/feature_engineering/feature_engineering.py tests/unit/test_feature_engineering.py
git commit -m "Declare the 16-channel feature vector in one place"
```

---

### Task 4: Targets carry only the forecast channels

**Files:**
- Modify: `jobs/model_training/data_loader.py:15, 128-136`
- Modify: `tests/unit/test_data_loader.py`

**Interfaces:**
- Consumes: `lakehouse.OUTPUT_CHANNELS` from Task 1.
- Produces: `IcebergTimeSeriesDataset.__getitem__` returning `x` of width `INPUT_CHANNELS` and `y` of width `OUTPUT_CHANNELS`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_data_loader.py`:

```python
import torch

from lakehouse import OUTPUT_CHANNELS


def test_the_target_carries_only_the_forecast_channels():
    """x keeps every input channel; y keeps only what the model predicts. Returning
    all 16 as the target would have the model fitting the calendar it was handed."""
    dataset = IcebergTimeSeriesDataset.__new__(IcebergTimeSeriesDataset)
    dataset.seq_len, dataset.pred_len = 3, 2
    dataset.valid_starts = np.array([0], dtype=np.int64)
    dataset.data = torch.arange(16 * 5, dtype=torch.float32).reshape(5, 16)

    x, y = dataset[0]

    assert x.shape == (3, 16)
    assert y.shape == (2, OUTPUT_CHANNELS)
    # y must be the first columns of the rows that follow x, not a reshaped slice
    assert torch.equal(y, dataset.data[3:5, :OUTPUT_CHANNELS])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker run --rm --user root -v "$PWD/tests":/tests -e PYTHONPATH=/app dag-pytorch-model-training:1.0 sh -c "pip install -q pytest && python -m pytest /tests/unit/test_data_loader.py -q -p no:cacheprovider"`

Expected: FAIL — `assert torch.Size([2, 16]) == (2, 4)`.

- [ ] **Step 3: Slice the target**

In `jobs/model_training/data_loader.py`, extend the import:

```python
from lakehouse import OUTPUT_CHANNELS, load_iceberg_catalog, scan_ordered
```

and in `__getitem__`:

```python
        x = self.data[start_idx:end_idx]
        # Only the weather channels are forecast. The calendar channels are
        # deterministic and the exogenous ones are inputs; asking the model to
        # reproduce them spends capacity on a task with no value.
        y = self.data[end_idx:target_end, :OUTPUT_CHANNELS]
        return x, y
```

- [ ] **Step 4: Run the tests** — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add jobs/model_training/data_loader.py tests/unit/test_data_loader.py
git commit -m "Forecast only the weather channels, not the calendar"
```

---

### Task 5: Models take 16 in and 4 out

**Files:**
- Modify: `jobs/model_training/train_lstm.py:49-53`
- Modify: `jobs/model_training/train_transformer.py:53-57`
- Modify: `tests/unit/test_contract.py`

**Interfaces:**
- Consumes: `CONFIG["input_dim"]`/`CONFIG["output_dim"]` from Task 1.
- Produces: both models accept `(batch, 72, 16)` and return `(batch, 24, 4)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_contract.py`:

```python
import torch

from models import ConvLSTMWeatherForecaster, TimeSeriesTransformer


def _batch():
    return torch.zeros(2, lakehouse.SEQ_LEN, lakehouse.INPUT_CHANNELS)


def test_both_architectures_map_the_input_width_to_the_output_width():
    """input_dim and output_dim used to be the same number passed twice. They are
    genuinely different now, and passing the input width as the output width would
    have the model emitting calendar predictions nobody reads."""
    expected = (2, lakehouse.PRED_LEN, lakehouse.OUTPUT_CHANNELS)

    lstm = ConvLSTMWeatherForecaster(
        lakehouse.INPUT_CHANNELS, 64, lakehouse.OUTPUT_CHANNELS, lakehouse.PRED_LEN, 0.2
    )
    transformer = TimeSeriesTransformer(
        lakehouse.INPUT_CHANNELS, 32, 2, 4, 128,
        lakehouse.OUTPUT_CHANNELS, lakehouse.PRED_LEN, 0.1
    )

    assert lstm(_batch()).shape == expected
    assert transformer(_batch()).shape == expected


def test_the_training_configs_agree_with_the_contract():
    import train_lstm
    import train_transformer

    for config in (train_lstm.CONFIG, train_transformer.CONFIG):
        assert config["input_dim"] == lakehouse.INPUT_CHANNELS
        assert config["output_dim"] == lakehouse.OUTPUT_CHANNELS
        assert config["seq_len"] == lakehouse.SEQ_LEN
        assert config["pred_len"] == lakehouse.PRED_LEN
```

- [ ] **Step 2: Run test to verify it fails**

Run the training-image pytest on `tests/unit/test_contract.py`.
Expected: FAIL — `KeyError: 'feature_dim'` raised while importing `train_lstm`.

- [ ] **Step 3: Fix the construction call sites**

In `jobs/model_training/train_lstm.py`:

```python
    model = ConvLSTMWeatherForecaster(
        CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"],
        CONFIG["pred_len"], CONFIG["dropout"]
    ).to(device)
```

In `jobs/model_training/train_transformer.py`:

```python
    model = TimeSeriesTransformer(
        CONFIG["input_dim"], CONFIG["d_model"], CONFIG["n_heads"], CONFIG["num_layers"],
        CONFIG["dim_feedforward"], CONFIG["output_dim"], CONFIG["pred_len"],
        CONFIG["dropout"]
    ).to(device)
```

- [ ] **Step 4: Run the tests** — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add jobs/model_training/train_lstm.py jobs/model_training/train_transformer.py tests/unit/test_contract.py
git commit -m "Separate the model input width from its output width"
```

---

### Task 6: Baselines forecast the output channels

**Files:**
- Modify: `jobs/model_training/baselines.py`
- Modify: `tests/unit/test_baselines.py`

**Interfaces:**
- Consumes: `lakehouse.OUTPUT_CHANNELS`.
- Produces: `persistence_forecast(x, pred_len, output_channels)` and `seasonal_naive_forecast(x, pred_len, output_channels)`; `NAIVE_FORECASTS` unchanged as a name→callable mapping.

- [ ] **Step 1: Rewrite the tests**

Replace the top of `tests/unit/test_baselines.py` (`SEQ, PRED, FEATURES` and `_context`) with:

```python
SEQ, PRED, INPUTS, OUTPUTS = 72, 24, 16, 4


def _context(batch=2):
    """Each hour carries its own index as its value, so a wrong slice is visible."""
    return torch.arange(SEQ, dtype=torch.float32).reshape(1, SEQ, 1).repeat(batch, 1, INPUTS)
```

and update every call and assertion to pass `OUTPUTS` and expect it:

```python
def test_persistence_repeats_the_last_observed_hour():
    out = persistence_forecast(_context(), PRED, OUTPUTS)
    assert out.shape == (2, PRED, OUTPUTS)
    assert torch.all(out == SEQ - 1)


def test_seasonal_naive_replays_the_same_hour_one_day_earlier():
    """Target hour t+k is predicted by hour t+k-24, which sits inside the context."""
    out = seasonal_naive_forecast(_context(), PRED, OUTPUTS)
    assert out.shape == (2, PRED, OUTPUTS)
    assert torch.all(out[:, 0, :] == SEQ - PRED)
    assert torch.all(out[:, -1, :] == SEQ - 1)


def test_the_two_baselines_are_not_the_same_forecast():
    """If they agreed there would be no point logging both."""
    x = _context()
    assert not torch.equal(
        persistence_forecast(x, PRED, OUTPUTS), seasonal_naive_forecast(x, PRED, OUTPUTS)
    )


def test_seasonal_naive_refuses_a_context_shorter_than_the_horizon():
    with pytest.raises(ValueError, match="context"):
        seasonal_naive_forecast(torch.zeros(1, 10, INPUTS), PRED, OUTPUTS)


def test_a_baseline_never_returns_the_input_width():
    """The context carries 16 channels and the target carries 4. Returning all of
    them makes the baseline uncomparable against y - loudly, but only at runtime."""
    x = _context()
    assert persistence_forecast(x, PRED, OUTPUTS).shape[-1] == OUTPUTS
    assert seasonal_naive_forecast(x, PRED, OUTPUTS).shape[-1] == OUTPUTS
```

- [ ] **Step 2: Run tests to verify they fail**

Run the training-image pytest on `tests/unit/test_baselines.py`.
Expected: FAIL — `TypeError: persistence_forecast() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Narrow the baselines**

In `jobs/model_training/baselines.py`:

```python
def persistence_forecast(x, pred_len, output_channels):
    """Every future hour equals the last observed hour.

    The classic "no skill" reference. On an hourly weather series it is a weak one -
    it throws the daily cycle away - but it is the floor any forecaster must clear.
    """
    return x[:, -1:, :output_channels].expand(-1, pred_len, -1)


def seasonal_naive_forecast(x, pred_len, output_channels):
    """Every future hour equals the same hour one day earlier.

    Target hour t+k is predicted by t+k-24, and for a 24-hour horizon all of those sit
    inside a 72-hour context. This is the baseline that matters on a strongly diurnal
    series: it reproduces the daily cycle for free, so beating it is the real test.
    """
    if x.shape[1] < pred_len:
        raise ValueError(
            f"seasonal-naive needs at least {pred_len} hours of context, got {x.shape[1]}."
        )
    return x[:, -pred_len:, :output_channels]
```

Add above `NAIVE_FORECASTS`:

```python
# The context carries every input channel; a forecast has to be the width of the
# target, or it cannot be scored against y at all.
```

- [ ] **Step 4: Run the tests** — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add jobs/model_training/baselines.py tests/unit/test_baselines.py
git commit -m "Baselines forecast the target width, not the context width"
```

---

### Task 7: Per-horizon error in the promotion benchmark

**Files:**
- Modify: `jobs/model_training/lakehouse.py`
- Modify: `jobs/model_training/evaluate_and_promote.py:27-90, 125-175`
- Modify: `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: `lakehouse.TEMPERATURE_CHANNEL`, `OUTPUT_CHANNELS`, `PRED_LEN`.
- Produces: `lakehouse.load_scaling_parameters(catalog) -> dict[str, tuple[float, float]]`; `Scores` gains `horizon_rmse` (np.ndarray of length `pred_len`, normalized units); `evaluate_and_promote.horizon_rmse_celsius(scores, temperature_std) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_evaluation.py`:

```python
import numpy as np

from evaluate_and_promote import horizon_rmse_celsius, score_predictors
from lakehouse import OUTPUT_CHANNELS


def _horizon_loader(n_batches=2, batch=4, seq=72):
    """Targets whose error grows with the horizon, by construction."""
    batches = []
    for _ in range(n_batches):
        x = torch.zeros(batch, seq, 16)
        y = torch.zeros(batch, PRED, OUTPUT_CHANNELS)
        batches.append((x, y))
    return CountingLoader(batches)


def _ramp(x):
    """Predicts hour k as k+1, so the error at horizon k is exactly k+1."""
    steps = torch.arange(1, PRED + 1, dtype=torch.float32)
    return steps.view(1, PRED, 1).expand(x.shape[0], PRED, OUTPUT_CHANNELS)


def test_per_horizon_rmse_is_reported_for_every_forecast_hour():
    scores = score_predictors({"ramp": _ramp}, _horizon_loader(), nn.SmoothL1Loss())
    horizon = scores["ramp"].horizon_rmse

    assert horizon.shape == (PRED,)
    # error at horizon k is k+1 by construction
    np.testing.assert_allclose(horizon, np.arange(1, PRED + 1), rtol=1e-5)


def test_per_horizon_rmse_converts_to_celsius_by_the_published_std():
    """The gate works in normalized units; the number a person reads has to be a
    temperature. scaling_parameters is what makes that conversion exact."""
    scores = score_predictors({"ramp": _ramp}, _horizon_loader(), nn.SmoothL1Loss())
    celsius = horizon_rmse_celsius(scores["ramp"], temperature_std=8.164)
    np.testing.assert_allclose(celsius, np.arange(1, PRED + 1) * 8.164, rtol=1e-5)


def test_a_flat_forecast_has_a_flat_horizon_curve():
    """Guards against accumulating across horizons instead of per horizon."""
    scores = score_predictors(
        {"off_by_one": lambda x: torch.ones(x.shape[0], PRED, OUTPUT_CHANNELS)},
        _horizon_loader(), nn.SmoothL1Loss(),
    )
    np.testing.assert_allclose(scores["off_by_one"].horizon_rmse, np.ones(PRED), rtol=1e-5)
```

Also add `PRED = 24` beside the existing constant at the top of the file if it is not already named that, and update `_loader`'s `feats` default from `4` to `16` with `y` sliced to `OUTPUT_CHANNELS`:

```python
def _loader(n_batches=3, batch=4, seq=72, feats=16, seed=0):
    """Batches whose target is the last PRED hours of their own context, narrowed to
    the forecast channels."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        x = torch.randn(batch, seq, feats, generator=generator)
        batches.append((x, x[:, -PRED:, :OUTPUT_CHANNELS].clone()))
    return CountingLoader(batches)


def _perfect(x):
    return x[:, -PRED:, :OUTPUT_CHANNELS]


def _off_by_one(x):
    return x[:, -PRED:, :OUTPUT_CHANNELS] + 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run the training-image pytest on `tests/unit/test_evaluation.py`.
Expected: FAIL — `ImportError: cannot import name 'horizon_rmse_celsius'`.

- [ ] **Step 3: Add the shared scaling reader**

Append to `jobs/model_training/lakehouse.py`:

```python
def load_scaling_parameters(catalog):
    """{feature_name: (mean, std)} exactly as 02 published them.

    The gate scores in normalized units, which are unreadable. This is the same table
    the serving API inverts with, so a number reported here and a number on the
    dashboard mean the same thing.
    """
    table = catalog.load_table(("weather", "scaling_parameters"))
    arrow = table.scan().to_arrow()
    return {
        name: (float(mean), float(std))
        for name, mean, std in zip(
            arrow.column("feature_name").to_pylist(),
            arrow.column("mean_value").to_pylist(),
            arrow.column("std_value").to_pylist(),
        )
    }
```

- [ ] **Step 4: Accumulate per horizon**

In `jobs/model_training/evaluate_and_promote.py`, add `horizon_rmse: "object"` to `Scores`, and in `_Running.__init__` add:

```python
        # Squared error per forecast hour for the temperature channel. Kept separate
        # from window_mse because the promotion test wants one number per window and
        # a person wants one number per horizon - the aggregate hides that error more
        # than doubles from t+1 to t+24.
        self.horizon_sq_error = torch.zeros(PRED_LEN)
        self.windows = 0
```

In the batch loop of `score_predictors`, after `acc.window_mse.append(...)`:

```python
                temp_error = diff[:, :, TEMPERATURE_CHANNEL]
                acc.horizon_sq_error += (temp_error * temp_error).sum(dim=0).cpu()
                acc.windows += temp_error.shape[0]
```

and in the returned `Scores`:

```python
            horizon_rmse=torch.sqrt(acc.horizon_sq_error / acc.windows).numpy(),
```

Add beneath `score_predictors`:

```python
def horizon_rmse_celsius(scores, temperature_std):
    """Per-horizon temperature RMSE in degrees, from the published standard deviation."""
    return scores.horizon_rmse * temperature_std
```

- [ ] **Step 5: Report it**

In `evaluate_and_promote()`, after the naive scores are computed, read the scaling
parameters and print the horizon table:

```python
    # Normalized RMSE is not a temperature. Read the same parameters the serving API
    # inverts with so a number here and a number on the dashboard mean the same thing.
    temperature_std = 1.0
    try:
        published = load_scaling_parameters(load_iceberg_catalog())
        temperature_std = published["temperature"][1]
        unit = "degC"
    except Exception as e:
        logger.warning(f"Could not read scaling parameters ({e}); horizons are in normalized units.")
        unit = "normalized"

    logger.info(f"--- ERROR BY HORIZON ({unit}, temperature) ---")
    for label, scores in [(f"champion v{champion_version}", champion),
                          (f"challenger v{challenger_version}", challenger)] + list(naive.items()):
        curve = horizon_rmse_celsius(scores, temperature_std)
        logger.info(
            f"  {label:<28} t+1={curve[0]:.3f}  t+6={curve[5]:.3f}  "
            f"t+12={curve[11]:.3f}  t+24={curve[-1]:.3f}  mean={curve.mean():.3f}"
        )
```

and inside the MLflow run, beside the other metrics:

```python
        # All 24, so the curve can be charted; the log above prints the four that get
        # read by eye.
        for label, scores in [("champion", champion), ("challenger", challenger)] + list(naive.items()):
            for hour, value in enumerate(horizon_rmse_celsius(scores, temperature_std), start=1):
                metrics[f"{label}_temp_rmse_h{hour:02d}"] = float(value)
```

Add `load_iceberg_catalog` and `load_scaling_parameters` to the `from lakehouse import ...` line.

- [ ] **Step 6: Run the tests** — expected PASS (all of `test_evaluation.py`).

- [ ] **Step 7: Commit**

```bash
git add jobs/model_training/lakehouse.py jobs/model_training/evaluate_and_promote.py tests/unit/test_evaluation.py
git commit -m "Report benchmark error by forecast horizon, in degrees"
```

---

### Task 8: A champion that predates the feature schema

**Files:**
- Modify: `jobs/model_training/evaluate_and_promote.py`
- Modify: `tests/unit/test_evaluation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `evaluate_and_promote.can_score(model, dataloader) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_evaluation.py`:

```python
from evaluate_and_promote import can_score


def test_a_model_that_matches_the_current_vector_can_be_scored():
    assert can_score(lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS), _loader())


def test_a_model_from_an_older_feature_schema_is_detected_not_raised():
    """Loading such a champion succeeds; the first forward pass is where the width
    mismatch surfaces. Letting it escape aborts the whole promotion task, which is
    how a schema change turns into a red DAG instead of a decision."""
    def four_channel_model(x):
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (4x16 and 4x32)")

    assert can_score(four_channel_model, _loader()) is False


def test_the_check_consumes_only_one_batch():
    """113k windows are not needed to learn that a model does not fit."""
    loader = _loader()
    can_score(lambda x: torch.zeros(x.shape[0], PRED, OUTPUT_CHANNELS), loader)
    assert loader.passes == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run the training-image pytest on `tests/unit/test_evaluation.py`.
Expected: FAIL — `ImportError: cannot import name 'can_score'`.

- [ ] **Step 3: Implement the check**

Add to `jobs/model_training/evaluate_and_promote.py`, beneath `score_predictors`:

```python
def can_score(predict, dataloader):
    """Whether this predictor still runs against the current feature vector.

    A champion registered before a feature-schema change loads fine - the artifact is
    intact - and fails on its first forward pass, because the input width moved under
    it. Detecting that here turns a red DAG into a decision.
    """
    x, _ = next(iter(dataloader))
    try:
        with torch.no_grad():
            predict(x.to(device))
        return True
    except Exception as e:
        logger.error(f"Model cannot be scored against the current feature vector: {e}")
        return False
```

- [ ] **Step 4: Wire it into the gate**

In `evaluate_and_promote()`, immediately after both models are loaded and set to
`eval()`, and before `score_predictors`:

```python
    # A champion from before a feature-schema change cannot be benchmarked at all.
    # Promote the challenger as a fresh baseline rather than failing the task: there
    # is no comparison to make, and the old model can no longer serve either.
    if not can_score(champion_model, test_loader):
        logger.error(
            f"Champion v{champion_version} predates the current feature vector and "
            f"cannot be scored. Promoting challenger v{challenger_version} as a new baseline."
        )
        client.set_registered_model_alias(name=model_name, alias="champion", version=challenger_version)
        # Deliberately not 'previous_champion': that alias implies something you could
        # roll back to, and this model cannot run.
        client.set_registered_model_alias(name=model_name, alias="retired_champion", version=champion_version)
        with mlflow.start_run(run_name=f"Eval_v{challenger_version}_schema_change"):
            mlflow.log_params({
                "model_name": model_name,
                "champion_version": champion_version,
                "challenger_version": challenger_version,
                "promotion_decision": "PROMOTED_SCHEMA_CHANGE",
            })
        return
```

- [ ] **Step 5: Run the tests** — expected PASS.

- [ ] **Step 6: Commit**

```bash
git add jobs/model_training/evaluate_and_promote.py tests/unit/test_evaluation.py
git commit -m "Promote around a champion the feature schema left behind"
```

---

### Task 9: Forecasts record their horizon

**Files:**
- Modify: `jobs/model_training/batch_inference.py:140-160, 210-265`
- Modify: `tests/unit/test_evaluation.py` is **not** the right home — create `tests/unit/test_batch_inference.py`
- Modify: `dev.sh`, `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `lakehouse.PRED_LEN`.
- Produces: `batch_inference.forecast_rows(model_name, version, future_timestamps, predictions, now)` returning a dict of column-name → list, including `horizon`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_batch_inference.py`:

```python
"""Row construction for the forecast table. Runs inside dag-pytorch-model-training."""
import datetime

from batch_inference import forecast_rows
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
```

- [ ] **Step 2: Run test to verify it fails**

Run the training-image pytest on `tests/unit/test_batch_inference.py`.
Expected: FAIL — `ImportError: cannot import name 'forecast_rows'`.

- [ ] **Step 3: Add the horizon to the table schema**

In `jobs/model_training/batch_inference.py`, add `IntegerType` to the pyiceberg type
import and extend the schema in `open_predictions_table`:

```python
        schema = Schema(
            NestedField(1, "forecast_timestamp", TimestamptzType(), required=True),
            NestedField(2, "predicted_features", ListType(element_id=3, element_type=FloatType()), required=True),
            NestedField(4, "model_name", StringType(), required=True),
            NestedField(5, "model_version", StringType(), required=True),
            NestedField(6, "created_at", TimestamptzType(), required=True),
            # 1-24. Known here and nowhere downstream, and without it every residual
            # is an average over horizons whose difficulty differs by a factor of two.
            NestedField(7, "horizon", IntegerType(), required=True),
        )
```

- [ ] **Step 4: Extract row construction**

Add above `main()`:

```python
def forecast_rows(model_name, version, future_timestamps, predictions, now_utc):
    """One row per predicted hour, as column-name -> list."""
    return {
        "forecast_timestamp": list(future_timestamps),
        "predicted_features": [list(row) for row in predictions],
        "model_name": [model_name] * PRED_LEN,
        "model_version": [str(version)] * PRED_LEN,
        "created_at": [now_utc] * PRED_LEN,
        "horizon": list(range(1, PRED_LEN + 1)),
    }
```

In `main()`, replace the five `all_*` accumulator lists with one:

```python
    columns = {
        "forecast_timestamp": [], "predicted_features": [], "model_name": [],
        "model_version": [], "created_at": [], "horizon": [],
    }
```

the per-model `extend` block with:

```python
            for key, values in forecast_rows(
                model_name, version, future_timestamps, predictions[0].tolist(), now_utc
            ).items():
                columns[key].extend(values)
```

the emptiness check with `if not columns["model_name"]:`, the Arrow schema with an
added field:

```python
        pa.field("horizon", pa.int32(), nullable=False),
```

and the `pa.Table.from_arrays` call with `pa.array(columns["horizon"], type=pa.int32())`
appended, every other array reading from `columns[...]`. The final log line's
`len(all_model_names)` becomes `len(columns["model_name"])`.

- [ ] **Step 5: Register the new test file in both runners**

Append `/tests/unit/test_batch_inference.py` to `dev.sh` and `tests/unit/test_batch_inference.py \` to `ci.yml`, both in the training block.

- [ ] **Step 6: Run the tests**

Run `python3 -m pytest tests/static -q` — expected PASS.
Run the training-image pytest on `tests/unit/test_batch_inference.py` — expected PASS.

- [ ] **Step 7: Commit**

```bash
git add jobs/model_training/batch_inference.py tests/unit/test_batch_inference.py dev.sh .github/workflows/ci.yml
git commit -m "Record the horizon each forecast row describes"
```

---

### Task 10: Serving reports error by horizon

**Files:**
- Modify: `jobs/serving/api.py:196-224`
- Modify: `jobs/serving/dashboard.py:38-64`
- Modify: `tests/unit/test_serving.py:58-62, 96`

**Interfaces:**
- Consumes: the `horizon` column from Task 9.
- Produces: `/api/v1/metrics/residuals` rows carrying `horizon`.

- [ ] **Step 1: Update the fixtures and write the failing test**

In `tests/unit/test_serving.py`, give `_forecast_rows` a horizon and widen the actuals:

```python
def _forecast_rows(hours, model, created_at, value=0.5, version="1"):
    return [{"forecast_timestamp": h, "predicted_features": [value, 0.0, 0.0, 0.0],
             "model_name": model, "model_version": version, "created_at": created_at,
             "horizon": i + 1}
            for i, h in enumerate(hours)]
```

and in `test_residuals_dedup_keeps_the_newest_write_whatever_the_scan_order`, the
actuals frame becomes a 16-channel vector, because that is what `ml_features` holds now:

```python
    actuals = pd.DataFrame([{"timestamp": h, "features": [1.0] + [0.0] * 15} for h in hours])
```

Append:

```python
def test_residuals_are_reported_per_horizon(monkeypatch, hours):
    """Averaging across horizons hides that t+24 error is more than twice t+1.
    One row per (model, version, horizon) is what makes the curve visible."""
    preds = pd.DataFrame(
        _forecast_rows(hours, "M", pd.Timestamp("2026-08-20 05:00", tz="UTC"), value=1.0)
    )
    actuals = pd.DataFrame([{"timestamp": h, "features": [1.0] + [0.0] * 15} for h in hours])
    _fake_tables(monkeypatch, forecast_predictions=preds, ml_features=actuals)

    metrics = api.get_residuals()

    assert len(metrics) == len(hours)
    assert sorted(row["horizon"] for row in metrics) == list(range(1, len(hours) + 1))
    assert all(row["samples"] == 1 for row in metrics)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker run --rm --user root -v "$PWD/tests":/tests -e PYTHONPATH=/app lakehouse-lakehouse-serving:latest sh -c "pip install -q pytest && python -m pytest /tests/unit/test_serving.py -q -p no:cacheprovider"`

Expected: FAIL — `KeyError: 'horizon'`.

- [ ] **Step 3: Group by horizon**

In `jobs/serving/api.py`, change the aggregation in `get_residuals`:

```python
        # Grouped by horizon as well as by model and version: t+1 and t+24 are not the
        # same forecasting problem, and averaging them together reports a number that
        # describes neither. Measured on the test block, t+24 error is more than twice
        # t+1's.
        metrics = merged.groupby(['model_name', 'model_version', 'horizon']).agg(
            mae=('abs_error', 'mean'),
            rmse=('sq_error', lambda x: math.sqrt(x.mean())),
            samples=('abs_error', 'size'),
        ).reset_index()

        return metrics.sort_values(['model_name', 'model_version', 'horizon']).to_dict(orient='records')
```

- [ ] **Step 4: Chart the curve**

In `jobs/serving/dashboard.py`, replace the two bar charts with error-versus-horizon
lines — a bar chart keyed on `model_name` would silently stack 24 rows per model:

```python
    if "status" not in metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics["model"] = df_metrics["model_name"] + " v" + df_metrics["model_version"].astype(str)
        col1, col2 = st.columns(2)

        for column, metric, title in ((col1, "mae", "Mean Absolute Error (MAE)"),
                                      (col2, "rmse", "Root Mean Square Error (RMSE)")):
            with column:
                fig = px.line(
                    df_metrics.sort_values("horizon"),
                    x="horizon", y=metric, color="model", markers=True,
                    labels={"horizon": "Forecast horizon (hours ahead)", metric: "°C"},
                    title=f"{title} by horizon", template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for actual weather data to overlap with predictions to calculate errors.")
```

- [ ] **Step 5: Run the tests** — expected PASS (11 tests).

- [ ] **Step 6: Commit**

```bash
git add jobs/serving/api.py jobs/serving/dashboard.py tests/unit/test_serving.py
git commit -m "Report and chart serving error by forecast horizon"
```

---

### Task 11: Documentation

**Files:**
- Modify: `README.md` — the data model table, "Feature normalization", "Model artifacts" if it mentions widths, and a new note under "Operations"
- Modify: `PROJECT_CONTEXT.md` — §4 data model, §5 conventions

- [ ] **Step 1: Update the README data model**

Replace the sentence describing the vector:

```markdown
`ml_features.features` is a `list<float32>` of sixteen values: ten standardized
weather variables, then three sin/cos pairs for wind direction, hour of day and day of
year. The first four are `temperature, humidity, precipitation, wind_speed`, in that
order and at those positions permanently — the serving API de-normalizes channel 0 as
temperature from an image that cannot import the layout at all, and `drift_monitor`
and `evaluate_and_promote` read it through `TEMPERATURE_CHANNEL`, which still only
agrees with the writer by convention.
`scaling_parameters` describes all sixteen; the six cyclical channels are published
with identity scaling because they are already bounded in [-1, 1] and are never
standardized.

Models take all sixteen channels in and forecast the first four.
```

- [ ] **Step 2: Add the horizon note to the README**

Under "Operations", add:

```markdown
- **Error is reported per horizon.** t+1 and t+24 are not the same problem: measured
  on the test block, the champions beat seasonal-naive by 65% at t+1 and by 9% at
  t+24. `evaluate_and_promote` logs the curve for both models and both baselines, and
  `/api/v1/metrics/residuals` returns one row per `(model, version, horizon)`. A
  single averaged number describes neither end of the horizon.
```

- [ ] **Step 3: Update `PROJECT_CONTEXT.md`**

In §4, replace the `list<float32>` sentence with the same description as Step 1. In §5,
add a bullet:

```markdown
- **The 16-channel layout is declared once per image.** `feature_engineering.py` owns
  the writer's copy (`SCALED_COLUMNS` + `CYCLICAL_CHANNELS`), `lakehouse.py` owns the
  reader's (`INPUT_CHANNELS`, `OUTPUT_CHANNELS`, `TEMPERATURE_CHANNEL`), and the two
  images agree through `weather.scaling_parameters`. Channels 0-3 never move.
```

- [ ] **Step 4: Run the static suite**

Run: `python3 -m pytest tests/static -q` — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md PROJECT_CONTEXT.md
git commit -m "Document the 16-channel vector and per-horizon reporting"
```

---

### Task 12: Migrate and run

Destructive and deliberate. `/mnt/c/lakehouse-backup-2026-08-22/` holds the pre-change state.

**Files:** none — this is an operation.

- [ ] **Step 1: Rebuild every image**

```bash
./dev.sh build-jobs && docker compose build
```

- [ ] **Step 2: Run the whole suite**

```bash
./dev.sh test
```

Expected: all suites pass. Do not continue past a failure.

- [ ] **Step 3: Drop the tables and the registered models**

With the stack up (`./dev.sh up`), run:

```bash
docker run --rm --network lakehouse-net \
  -e AWS_ACCESS_KEY_ID="$(grep '^MINIO_APP_USER' .env | cut -d= -f2)" \
  -e AWS_SECRET_ACCESS_KEY="$(grep '^MINIO_APP_PASSWORD' .env | cut -d= -f2)" \
  dag-pytorch-model-training:1.0 python -c "
from lakehouse import load_iceberg_catalog
import mlflow
from mlflow.tracking import MlflowClient

catalog = load_iceberg_catalog()
for table in ('observations', 'ml_features', 'scaling_parameters', 'forecast_predictions'):
    try:
        catalog.drop_table(('weather', table)); print('dropped', table)
    except Exception as e:
        print('skip', table, e)

mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()
for name in ('Weather_Forecaster_FastLSTM', 'Weather_Forecaster_Transformer'):
    try:
        client.delete_registered_model(name); print('deleted', name)
    except Exception as e:
        print('skip', name, e)
"
```

- [ ] **Step 4: Backfill**

```bash
docker exec airflow-scheduler airflow dags trigger 01_extract_weather_data
```

Wait for `01` to reach `success` (roughly 11 minutes), then confirm `02` was
Dataset-triggered and succeeded, and that `04` ran and reported no champions.

- [ ] **Step 5: Train both architectures from scratch**

```bash
docker exec airflow-scheduler airflow dags trigger 03a_train_pytorch_lstm \
  --conf '{"TRAINING_MODE":"SCRATCH"}'
docker exec airflow-scheduler airflow dags trigger 03b_train_pytorch_transformer \
  --conf '{"TRAINING_MODE":"SCRATCH"}'
```

Both take the SCRATCH path against an empty registry and auto-promote as initial
baselines. The `single_gpu` pool serializes them.

- [ ] **Step 6: Train both architectures a second time, so the benchmark actually runs**

```bash
docker exec airflow-scheduler airflow dags trigger 03a_train_pytorch_lstm \
  --conf '{"TRAINING_MODE":"SCRATCH"}'
docker exec airflow-scheduler airflow dags trigger 03b_train_pytorch_transformer \
  --conf '{"TRAINING_MODE":"SCRATCH"}'
```

Step 5's run registered v1 against an empty registry, so `evaluate_and_promote` took
its initial-baseline early return (champion_version is None) and never reached the
benchmark - there was no champion yet to compare against, and the `ERROR BY HORIZON`
table Step 8 reads is only printed from inside that benchmark. This run registers v2
with v1 already `@champion`, so the gate scores both against the held-out test block
and actually prints the table that is the sole verdict on this whole change.

- [ ] **Step 7: Produce forecasts**

```bash
docker exec airflow-scheduler airflow dags trigger 04_batch_inference_pipeline
```

- [ ] **Step 8: Record the result**

Read the `ERROR BY HORIZON` block from each `evaluate_and_promote_model` task log and
compare against the pre-change numbers, which are the reference this whole change is
judged on:

| horizon | FastLSTM@v6 | Transformer@v3 | seasonal-naive |
|---|---|---|---|
| t+1 | 0.874 °C | 0.913 | 2.524 |
| t+12 | 1.903 | 1.924 | 2.524 |
| t+24 | 2.303 | 2.305 | 2.524 |
| mean | 1.873 | 1.881 | 2.524 |

Append the new table and a short verdict to `PROJECT_CONTEXT.md` §11, and commit.

---

## Self-Review

**Spec coverage.** Feature vector → Tasks 3, 4, 5. New ETL variables and DQ bounds →
Task 2. Per-horizon metrics → Tasks 7 (gate) and 10 (serving). Incompatible champion →
Task 8. Migration → Task 12. Documentation → Task 11. The `load_published_stats` zip
trap called out in the spec → Task 3 Step 5. The six audit findings: dashboard → Task
10; ETL positional indexing → Task 2; scattered channel knowledge → Task 3; duplicated
`SEQ_LEN`/`PRED_LEN` → Task 1; drift monitor → **gap**, addressed below; test fixtures
→ Tasks 6, 7, 10.

**Gap found and closed:** the drift monitor reads `features[0]` as temperature and
scans every column. Folded into Task 7 as an addendum:

- [ ] **Task 7, Step 8: Bound the drift monitor's read and name its assumption**

In `jobs/model_training/drift_monitor.py`, the main scan becomes:

```python
    df = table.scan(
        row_filter=climatology_filter(anchor),
        selected_fields=("timestamp", "features"),
    ).to_arrow().to_pandas()
```

and the temperature extraction gets its contract written down:

```python
    # Channel 0 is temperature by the layout 02 publishes to scaling_parameters, which
    # is fixed and which nothing here can verify - this image and that one never meet.
    df['temperature'] = df['features'].apply(lambda x: x[TEMPERATURE_CHANNEL])
```

with `TEMPERATURE_CHANNEL` added to its `from lakehouse import ...`. Extend
`tests/unit/test_drift_monitor.py` with a test that the scan is field-limited, and
include `jobs/model_training/drift_monitor.py` in Task 7's commit.

**Placeholder scan:** none found.

**Type consistency:** `score_predictors(predictors, dataloader, criterion) -> dict[str, Scores]`
is used identically in Tasks 7 and 8. `Scores.horizon_rmse` is produced in Task 7 and
consumed by `horizon_rmse_celsius` in the same task. `forecast_rows` (Task 9) returns
column-name → list and is consumed only by `main()` in the same task.
`persistence_forecast`/`seasonal_naive_forecast` gain a third parameter in Task 6; the
`NAIVE_FORECASTS` call site in `evaluate_and_promote.py` passes `PRED_LEN, OUTPUT_CHANNELS`
— **Task 6 Step 3 must also update that call site**, which reads
`lambda x, f=fn: f(x, PRED_LEN)` and becomes `lambda x, f=fn: f(x, PRED_LEN, OUTPUT_CHANNELS)`.
