# Weather Lakehouse

[![CI](https://github.com/efkirmizi/weather-lakehouse/actions/workflows/ci.yml/badge.svg)](https://github.com/efkirmizi/weather-lakehouse/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end lakehouse + MLOps stack that runs on a single machine: hourly weather
observations for one location land in Apache Iceberg, PySpark turns them into
normalized ML features, two PyTorch architectures compete for a `@champion` alias in
the MLflow registry, and the winning models produce 24-hour forecasts served through
FastAPI and Streamlit.

Scope is deliberately a single site (see [Known limitations](#known-limitations));
the name says lakehouse rather than pipeline because the catalog, serving and model
registry layers are part of it, not just the ETL.

The champion forecasts a full day ahead **16.5% better than a seasonal baseline**, and
the promotion gate that decides which model serves has rejected a plausible change on
the evidence — see [Results](#results) for both experiments, including the one that
failed.

**Contents**

| | |
|---|---|
| **What it does** | [Stack](#stack) · [Results](#results) · [Running it](#running-it) · [Pipeline](#pipeline) |
| **Data** | [Data model](#data-model) · [Feature normalization](#feature-normalization) · [Sequence ordering](#sequence-ordering) · [Timestamps](#timestamps) |
| **Modelling** | [Train / validation / test / adapt](#train--validation--test--adapt) · [Promotion](#promotion) · [Drift detection](#drift-detection) · [Model artifacts](#model-artifacts) |
| **Running it for real** | [Forecast idempotency](#forecast-idempotency) · [Maintenance and garbage collection](#maintenance-and-garbage-collection) · [Operations](#operations) |
| **Engineering** | [Tests](#tests) · [Known limitations](#known-limitations) · [Project status](#project-status) |

Most sections explain a decision and the failure that motivated it, rather than
restating what the code does. If you only read three, read
[Results](#results), [Promotion](#promotion) and
[Train / validation / test / adapt](#train--validation--test--adapt) — that is where
the reasoning that matters lives.

## Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow 2.10.1 — LocalExecutor, Dataset-driven scheduling, DockerOperator |
| Object storage | MinIO (`warehouse` and `mlflow` buckets) |
| Table format | Apache Iceberg 1.5.0 |
| Catalog | Project Nessie 0.108.4, JDBC2 version store on PostgreSQL |
| Processing | PySpark 3.5.2 + Spark MLlib; PyIceberg on the Python side |
| Training | PyTorch 2.7.1 / CUDA 11.8 |
| Serving graph | ONNX Runtime (CPU) |
| MLOps | MLflow 3.15.1 — tracking + Model Registry with alias-based promotion |
| Serving | FastAPI (`:8000`) + Streamlit/Plotly (`:8501`) |

## Results

Every number here is measured on the same held-out block: 113,856 windows the models
were never fitted or early-stopped on.

The current per-horizon figures and the capacity experiment below come straight from
`evaluate_and_promote`, the same code path that decides deployments. The
before/after table for the feature change was produced by a standalone script, because
the run that would have gated it lost its artifact upload to a disk fault — but its
"after" column was later reproduced digit for digit by the gate itself, which is the
only reason it is quoted here.

### Forecast skill

Temperature RMSE in °C, per forecast horizon. Lower is better.

| horizon | Transformer | FastLSTM | seasonal-naive | persistence | Transformer skill |
|---|---|---|---|---|---|
| t+1  | **0.813** | 0.868 | 2.524 | 0.982 | **67.8%** |
| t+3  | **1.124** | 1.146 | 2.524 | — | **55.5%** |
| t+6  | **1.437** | 1.432 | 2.524 | 4.192 | **43.1%** |
| t+12 | **1.782** | 1.825 | 2.524 | 5.591 | **29.4%** |
| t+18 | **1.961** | 1.998 | 2.524 | — | **22.3%** |
| t+24 | **2.108** | 2.175 | 2.524 | 2.524 | **16.5%** |

Skill is against `seasonal_naive`, the harder of the two baselines over the full
horizon. Both baselines are reported because each is only hard in one place:
persistence wins at t+1 and collapses by t+6, while seasonal-naive is flat by
construction — it predicts the same hour yesterday, so its error does not depend on
how far ahead you ask. They meet at exactly t+24, where "the last observed hour" and
"the same hour yesterday" are the same reading.

**The shape of that column is the honest headline.** Forecasting the next hour is
nearly free; forecasting a full day ahead is where a model earns its keep, and 16.5%
over a seasonal baseline at t+24 is a real but modest margin. A single averaged RMSE
would have hidden both facts.

### Two experiments, one positive and one negative

The pipeline exists to answer "does this change help?" with evidence. It has been
asked twice, and the second answer was no.

**1. More inputs — the feature vector went from 4 channels to 16.** Ten standardized
weather variables (temperature, humidity, precipitation, wind speed, pressure, dew
point, cloud cover, shortwave radiation, soil temperature, soil moisture) plus sin/cos
pairs for wind direction, hour of day and day of year. Tables dropped and rebuilt from
1940, both models retrained from scratch:

| | before | after | change |
|---|---|---|---|
| FastLSTM, mean RMSE | 1.873 | **1.762** | −5.9% |
| Transformer, mean RMSE | 1.881 | **1.724** | −8.3% |
| Transformer skill at t+24 | 8.7% | **16.5%** | nearly doubled |

Every horizon improved for both architectures. The architectures also separated: they
had previously landed within 0.4% of each other, and the Transformer turned out to use
the extra inputs better.

The design record for this change is in
[`docs/design/`](docs/design/2026-08-22-exogenous-and-calendar-features.md) and the
task breakdown in [`docs/plans/`](docs/plans/2026-08-22-exogenous-and-calendar-features.md)
— written before the code, including the channel layout the three images still agree on
today.

**2. A bigger model — the Transformer's width was scaled 3.8×.** `d_model` 32 → 64,
`n_heads` 2 → 4, `dim_feedforward` 128 → 256, depth held at 4 so the experiment read as
width alone. 54,528 → 207,264 parameters. The gate **rejected it**:

| horizon | champion (d_model 32) | challenger (d_model 64) |
|---|---|---|
| t+1  | **0.813** | 0.933 |
| t+6  | **1.437** | 1.488 |
| t+12 | **1.782** | 1.807 |
| t+24 | **2.108** | 2.186 |

`PROMOTION REJECTED ... improvement -0.274% is below the 0.1% floor.` Worse at every
horizon, not just on the summary. The training curve had already said so: validation
loss bottomed at epoch 4 and rose for three consecutive epochs while training loss kept
falling. Early stopping kept epoch 4's weights and epoch 4's weights still lost.

**Read together: this problem is limited by what the model is told, not by how much
model there is.** Four times the input bought −8.3%; four times the parameters bought
−0.3% in the wrong direction. The rejected configuration was reverted in the same pass,
because leaving it would have made every weekly incremental run silently fall back to a
full ten-epoch retrain.

### Why these numbers are comparable

Points of method that make the two tables above mean something:

- **The baselines are the control.** `seasonal_naive` reads 2.524 in every table on
  this page. It cannot be affected by a model change, so a run where it moved would
  mean the test block moved, and the comparison would be void.
- **The gate scores in fp32**, not mixed precision. The margins it decides on are
  ~0.2%, well inside AMP noise.
- **The test block is never trained or early-stopped on.** A fourth `adapt` block
  holds out the newest 14 days separately, so an incremental run can learn from recent
  data without touching what the gate scores.
- **Significance is tested on a subsample.** Consecutive windows share 95 of their 96
  hours, so the paired t-test takes one window per horizon — ~1,186 that share no
  timestep — rather than reporting overwhelming significance for anything at all.
- **Both models and both baselines are scored in a single walk** of the test block, on
  identical windows by construction.

## Running it

```bash
cp .env.example .env      # fill in every value, including DOCKER_GID

# The job images are launched by DockerOperator, not by Compose, so build them first.
docker build -t dag-pyspark-etl:1.0                 jobs/weather_etl
docker build -t dag-pyspark-feature-engineering:1.0 jobs/feature_engineering
docker build -t dag-pytorch-model-training:1.0      jobs/model_training
docker build -t dag-lakehouse-janitor:1.0           jobs/maintenance
docker build -t dag-nessie-gc:1.0                  jobs/nessie_gc

# Compose builds and starts everything else.
docker compose up -d --build
```

`./dev.sh` wraps all of that:

```bash
./dev.sh build     # job images + Compose services
./dev.sh up        # start the stack
./dev.sh test      # static checks + DAG parsing + unit tests
./dev.sh test-integration   # against the running stack, so 'up' first
```

`DOCKER_GID` comes from `getent group docker | cut -d: -f3`. The Airflow **scheduler**
needs it to reach `/var/run/docker.sock` and launch job containers; the webserver
deliberately does not get the socket.

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 |
| Nessie | http://localhost:19120 |
| Forecast API | http://localhost:8000 (`/health`, `/api/v1/forecast/latest`, `/api/v1/metrics/residuals`) |
| Dashboard | http://localhost:8501 |

On a cold start the DAGs run themselves in order — `01` is `@daily` and everything
downstream is triggered by Iceberg Datasets. The first `01` run backfills from
1940-01-01 in five-year chunks and takes a while.

**Training is the exception.** `03a` and `03b` are on a weekly cron, so a freshly
created stack has no registered model until the next Sunday 02:00. Until one exists,
`04` writes no forecasts and the dashboard has nothing to chart — both endpoints
report the empty state rather than an error. Trigger a training DAG by hand if you do
not want to wait for the cron.

To smoke-test a pipeline change without waiting for a full training budget, trigger a
training DAG with a run config:

```bash
airflow dags trigger 03b_train_pytorch_transformer \
  --conf '{"TRAINING_MODE":"SCRATCH","TRAINING_EPOCHS":"1"}'
```

`TRAINING_MODE=SCRATCH` overrides the registry branch (this is also how the drift
monitor forces a from-scratch retrain), and `TRAINING_EPOCHS` overrides the default
budget of 10 epochs from scratch / 2 incremental. Both are per-run; neither changes
the defaults.

## Pipeline

```
Open-Meteo Archive API
   │  01_extract_weather_data     @daily, watermarked chunks + data-quality gate
   ▼
weather.observations ──Dataset──► 02_precompute_ml_features
                                    appends only new hours; full rebuild when
                                    the normalization drifts (see below)
                                         │
                              weather.ml_features
                                    │                    │
                     ┌──────Dataset─┘                    └── 03a / 03b  weekly, Sun 02:00
                     ▼                                        branch on registry state:
          04_batch_inference_pipeline                          ├─ absent  → full training
            monitor_concept_drift (KS-test)                    └─ present → incremental
                     │  on drift: triggers 03a AND 03b               │
                     │  with conf TRAINING_MODE=SCRATCH              ▼
                     ▼                                       evaluate_and_promote
            generate_multi_model_forecasts                    paired test on the TEST
            every @champion, via ONNX Runtime                 split, vs naive baselines
            (skipped if already forecast)                     → new @champion
                     ▼
        weather.forecast_predictions ──► FastAPI ──► Streamlit

        05_lakehouse_maintenance   weekly, Sun 03:00
        data file compaction + manifest rewrite

        06_nessie_gc               weekly, Sun 12:00, SHIPPED PAUSED
        sweeps files no Nessie reference points at
```

## Data model

Nessie branch `main`, namespace `weather`:

| Table | Written by | Mode |
|---|---|---|
| `observations` | `01` | append |
| `ml_features` | `02` | append (full overwrite on rebuild) |
| `scaling_parameters` | `02` | overwrite, on rebuild only |
| `forecast_predictions` | `04` | append, partitioned by forecast day |

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

## Feature normalization

`02` standardizes each of the ten weather-variable channels with a single global mean
and standard deviation, and publishes them to `weather.scaling_parameters` so the
serving API can invert the transform back into degrees Celsius.

Because every row in `ml_features` has to share one normalization, the job does not
recompute the parameters on every run. It compares the current global statistics
against the published ones and takes one of two paths:

- **Incremental** (the normal case): only observations newer than the table's
  watermark are standardized, using the *published* parameters, and appended.
  `scaling_parameters` is left untouched.
- **Full rebuild**: the whole table is renormalized and the parameters republished.
  Triggered when the table or the parameters do not exist yet, when a run passes
  `conf {"FEATURE_REBUILD": "full"}`, when the statistics have drifted by more than
  `FEATURE_REBUILD_TOLERANCE` (default 1%), or when the row counts disagree — every
  observation up to the watermark must already have a feature row, which is what
  catches a row backfilled *behind* the watermark that the incremental path would
  otherwise skip forever.

Drift is measured as a fraction of the feature's own standard deviation rather than
of its mean, so a near-zero mean like precipitation does not force a rebuild every
time it rains.

## Train / validation / test / adapt

`split_windows` cuts the timeline into four chronological blocks — oldest to newest —
and drops the windows that straddle each cut, because consecutive windows share all
but one of their timesteps and a random split would hand the later blocks
near-duplicates of the training rows.

| Block | Size | Used by |
|---|---|---|
| train | 70% | gradient updates on a scratch run; the replay buffer on an incremental one |
| validation | 15% | early stopping and `val_*` metrics, in both modes |
| test | 15% | the promotion gate, and nothing else |
| adapt | newest 14 days | incremental runs, and nothing else |

The percentages are of the timeline *minus* the adapt block. Carving those newest
hours out first is what lets two conflicting requirements coexist: an incremental run
exists to learn from the newest data, and the gate has to score on windows that
neither model was fitted on.

Splitting the incremental *selection* instead — the original approach — looks
equivalent and is not. The recent windows sort last, so all 336 of them fell into the
val and test slices that training discards: an incremental run fitted nothing but its
historical replay buffer, while 156 of its validation windows sat inside the block the
gate scores on. Both `_select_indices` and `chronological_split` were individually
correct; only their composition was wrong, which is why the tests now assert on
`plan_windows` rather than on either half.

The gate needs its own block: a challenger early-stopped on the validation split
would otherwise be benchmarked on the data it was tuned against. `evaluate_and_promote`
also runs in fp32 — the margins it decides on are ~0.2%, well inside mixed-precision
noise, and a benchmark that decides deployments has to be reproducible.

Validation is deliberately the same block in both modes. It keeps `val_*` comparable
between runs, and scoring an incremental run against held-out history is exactly how
the catastrophic forgetting its replay buffer guards against would surface.

Windows that span a gap in the hourly series are skipped entirely. The ETL only warns
about gaps (the Open-Meteo archive genuinely has holes in the older decades), so
`IcebergTimeSeriesDataset` filters them and `batch_inference` refuses to forecast from
a discontinuous context window.

## Promotion

A challenger takes the `@champion` alias when its improvement is **worth deploying for
and unlikely to be noise** — two separate questions, asked separately.

| Check | Default | Why |
|---|---|---|
| relative RMSE improvement | ≥ 0.1% | below this it is not worth a deployment however reliable |
| paired one-sided t-test | p < 0.05 | the improvement has to survive being tested window by window |

This replaced a flat "challenger RMSE ≤ 99% of champion". That threshold was
unreachable: a warm-started incremental run improves by ~0.2%, so once warm-starting
began working the gate would have rejected every challenger it ever saw and
`@champion` would have frozen at the last from-scratch run. A fixed percentage also
asks the wrong question — it waves through a large gap that is pure noise and blocks a
small one that is perfectly consistent.

The test is paired because both models are scored on identical windows, and it is run
on a **subsample**: consecutive test windows share all but one of their 96 hours, so
~113k windows carry nowhere near 113k independent observations. Taking one window per
horizon leaves ~1,186 that share no timestep. Feeding the whole block to a t-test would
report overwhelming significance for anything at all.

Every decision is also logged against two forecasts that need no training —
`persistence` (every future hour equals the last observed one) and `seasonal_naive`
(every future hour equals the same hour yesterday). They do not gate anything; they are
what makes the champion's RMSE readable. A model that cannot beat seasonal-naive on an
hourly weather series is not forecasting, it is reproducing the diurnal cycle badly.

All four are scored by `score_predictors` in a **single** walk of the test block. They
see identical windows by construction, so a pass per predictor would have quadrupled
the cost of a task bounded at 15 minutes for no added information.

## Forecast idempotency

`04` is Dataset-triggered and `02` emits its outlet even on a no-op run, so the same
24 hours would otherwise be re-predicted from the same context on every trigger.
`batch_inference` checks `forecast_predictions` for the target window first and skips
any `(model_name, model_version)` already covered. Without this the table grew a fresh
copy per trigger and `/api/v1/metrics/residuals` averaged over all of them.

## Drift detection

`drift_monitor` compares the last 24 hours against the **same time of year in previous
years** (±7 days of the current day-of-year, over `DRIFT_BASELINE_YEARS`, default 30 —
the WMO climate-normal convention), not against the previous 30 days: on a strongly
seasonal hourly series the latter flags every ordinary seasonal transition. Bounding it
by year also keeps the read to a few thousand rows instead of the whole table.
A firing triggers a from-scratch retrain of every DAG in `DRIFT_RETRAIN_DAGS`, subject
to `DRIFT_COOLDOWN_HOURS` (default 168) so a warm spell cannot retrain daily.

## Model artifacts

Each training run logs three things to MLflow:

- **the registered model** — the float PyTorch module. This is what `@champion`
  points at, what incremental runs warm-start from, and what the promotion gate
  benchmarks.
- **`onnx_model`** — the same weights as an ONNX graph with a dynamic batch axis.
  `batch_inference.py` executes this through ONNX Runtime.
- **`quantized/`** — a dynamic INT8 `state_dict`, logged for size comparison only.
  It is deliberately not registered: dynamic quantization is CPU-only and its
  `state_dict` keys do not map back onto the float architecture.

## Maintenance and garbage collection

`05_lakehouse_maintenance` compacts data files and rewrites manifests. It does **not**
expire snapshots or sweep orphan files, and it should not be made to: Nessie sets
`gc.enabled=false` on every table it manages, so Iceberg rejects both procedures with

```
Cannot expire snapshots: GC is disabled (deleting files may corrupt other tables)
```

That is correct behaviour, not a misconfiguration. In a Git-like catalog the same data
file can be referenced from several branches and tags, so a file that looks unreachable
from `main` may still be live elsewhere.

Compaction alone is worth running: on the first pass it took `weather.observations`
from 76 data files to 1, with the row count unchanged.

Reclaiming storage is `06_nessie_gc`'s job. It runs
[`nessie-gc`](https://projectnessie.org/nessie-latest/gc/), the only tool that computes
the live set across *every* reference before sweeping. It is the one job here that
permanently deletes data, so it **ships paused** — unpause it deliberately.

`GC_CUTOFF` decides how much it removes:

| Cutoff | Effect |
|---|---|
| `NONE` (default) | No snapshot is expired. Still sweeps files no commit references at all — leftovers from writes that failed before committing. |
| `PT720H`, `100`, an ISO instant | Snapshots older than the cutoff are expired and their files deleted permanently. |

Override per run with `conf {"GC_CUTOFF": "PT720H"}`, and optionally set
`GC_MAX_FILE_MODIFICATION` to an ISO instant to protect anything written after it.

## Sequence ordering

Every consumer that slices sequences positionally goes through
`lakehouse.scan_ordered()`, which sorts by `timestamp`. Iceberg makes no ordering
guarantee across data files, and `05`'s `rewrite_data_files` can reshuffle them — so
reading rows in scan order would silently corrupt every training window and every
"last 72 hours" context. Do not bypass that helper.

## Tests

**151 tests across four suites.** There is no test-only image: each suite runs inside
the image that already carries its dependencies, with `tests/` mounted.

| Suite | Count | Where | What it protects |
|---|---|---|---|
| `tests/static` | 6 | any Python 3.11 | DAG discoverability, function-local import shadowing, suite registration, syntax. No dependencies, instant. |
| `tests/airflow` | 5 | the Airflow image | Parses `dags/` exactly as the scheduler does: import errors, one DAG per file, failure callbacks, GPU tasks inside the pool, deprecated arguments. |
| `tests/unit` | 124 | training / ETL / feature-engineering / serving images | Window splitting and its *composition* into a run's train/val/test sets, gap filtering, replay caps, warm-start version selection and architecture-mismatch fallback, MLflow client configuration, epoch overrides, single-pass benchmark scoring, the ETL watermark guard, the drift anchor probe, rebuild-vs-append decisions, forecast/residual selection, timestamp literals. |
| `tests/integration` | 16 | the training image, on the Compose network | The boundaries every unit suite has to stub: Iceberg through Nessie onto MinIO, `scan_ordered` against real out-of-order commits, the `scaling_parameters` layout three images agree on without being able to import each other, the champion's ONNX graph loaded through the URI batch inference uses, and the served forecast. |

The integration suite is the one that cannot be faked. `scaling_parameters` is written
by the feature-engineering image and read by the training and serving images, and none
of the three can import the others — so a unit test that stubs the table is only
asserting that one file agrees with itself. Nothing raises if the channels shift: the
API would de-normalize channel 0 with whatever parameters sit at index 0 and draw the
result on a chart labelled "temperature".

It needs the stack running, so it is `./dev.sh test-integration` rather than part of
`./dev.sh test`, which stays hermetic and instant.

Each static check exists because that failure actually happened here. A DAG file
that mentions neither "airflow" nor "dag" is skipped by `DAG_DISCOVERY_SAFE_MODE`
**with no import error** — the DAG just disappears. An `import mlflow.pytorch`
inside an `except` block makes `mlflow` function-local, so an earlier
`mlflow.onnx.load_model(...)` in the same function raises `UnboundLocalError` and
silently disables the whole ONNX serving path. And because `dev.sh` and `ci.yml`
enumerate the unit test files they run by hand, a new one is executed nowhere until
both lists are updated — which is how the training suites spent their whole life
running on a single laptop.

The DAG suite fails on any deprecation warning raised from `dags/`, because a
deprecated argument is not an error yet. `auto_remove=True` was the live case:
providers-docker 3.13.0 converted the bool to `'success'` and warned, so asserting
on `task.auto_remove` would have seen nothing wrong — the warning was the only
signal, and a later release turns it into a parse-time `ValueError`.

CI (`.github/workflows/ci.yml`) runs the static checks with no Docker at all, then
builds the light images for the rest. The training suites are the exception and run on
plain CPU wheels: the 11 GB CUDA image does not fit a hosted runner, but nothing they
assert — window planning, epoch resolution, registry selection, the drift anchor —
touches a GPU or a live MLflow server. That job is where every regression test for the
split and warm-start bugs lives, so it is the one that must not be dropped.

## Operations

- **Credentials.** The DockerOperator jobs get their S3 keys from the `minio_s3`
  Airflow connection, rendered per task run (`dags/credentials.py`). The scheduler and
  webserver no longer carry them. Set `AIRFLOW_FERNET_KEY` to encrypt them at rest.
- **Failure alerts.** Every DAG carries `on_failure_callback` from `dags/alerting.py`:
  a structured ERROR log always, plus a POST when `ALERT_WEBHOOK_URL` is set.
- **GPU serialization.** This host has one GPU and every GPU task requests all of
  them, so the `single_gpu` pool has **one** slot — `03a` and `03b` share a cron and
  would otherwise land on the same card together.
- **Stale-data guard.** Training refuses to run when the newest `ml_features` row is
  older than `MAX_FEATURE_AGE_HOURS` (default 72). Training is on a weekly cron, so
  without this a broken ETL produces a confident model trained on last week's world.
- **Maintenance fan-out.** `05` maps one task per table, so a failure on one does not
  force re-running compaction on the others.
- **Spark JARs are baked, not resolved.** The Iceberg and Nessie JARs live in
  `$SPARK_HOME/jars` in each Spark image. None of the three jobs sets
  `spark.jars.packages`: that config makes Spark run an Ivy resolution and re-download
  every JAR on each run whatever is already on the classpath, which cost ~200MB per
  container — `02` is dataset-triggered on every `01` and its container is
  auto-removed, so the cache was cold every time — and made Maven Central a hard
  runtime dependency. Bump the versions in the Dockerfiles.
- **Error is reported per horizon.** t+1 and t+24 are not the same problem: the
  Transformer champion beats seasonal-naive by 67.8% at t+1 and by 16.5% at t+24 (see
  [Results](#results)). `evaluate_and_promote` logs the curve for both models and both
  baselines, and `/api/v1/metrics/residuals` returns one row per
  `(model, version, horizon)`. A single averaged number describes neither end of the
  horizon, and averaging over horizons is how a residual chart hides that t+24 is three
  times harder than t+1.

## Timestamps

Every table stores `timestamptz`, and every value is UTC. `forecast_predictions` used
to store a naive `timestamp` because pyiceberg created it from an explicit schema
while Spark created the others — the values agreed, but pyiceberg rejects a row-filter
literal whose zone-awareness differs from the column, so anything filtering across
both tables had to special-case them. Keep new tables `timestamptz`.

## Known limitations

- **Single location, by choice.** `weather_etl.py` fetches one fixed coordinate
  (40.98, 27.51) and no table carries a location column. Going multi-site is a schema
  change plus a backfill, and it would also force a modelling decision — one global
  model with a location embedding, or one model per site and a registry entry for
  each. Not a config tweak.
- **The scheduler holds the Docker socket.** That is inherent to driving jobs with
  `DockerOperator`; any DAG author effectively has root on the host. Acceptable for a
  local stack, not for a shared one.
- **No auth on Nessie or MLflow**, and MLflow's backend store is SQLite. Both are
  fine for single-node development and neither is fine beyond it.
- **The integration suite needs the stack up, so CI never runs it.** `tests/integration`
  covers the boundaries the unit suites can only stub - Iceberg through Nessie onto
  MinIO, the `scaling_parameters` layout three images agree on without being able to
  import each other, the champion's ONNX graph and the served forecast. It needs
  Nessie, MinIO, MLflow and the API running and the 11 GB training image to run in, so
  it is `./dev.sh test-integration` after `./dev.sh up`, never a CI job. Spark is still
  covered only by running the DAGs.

## Project status

**Complete, and deliberately not left running.**

The pipeline has run end to end, the two questions it was built to answer are recorded
under [Results](#results) — one change accepted, one rejected — and 151 tests cover the
paths that produced them. There is no long-lived deployment behind this repository and
none is needed: the stack rebuilds from nothing with `./dev.sh build && ./dev.sh up`,
which was verified from an empty machine, and the warehouse refetches itself from the
Open-Meteo archive back to 1940.

That is the design working as intended rather than a caveat. Everything with state —
images, volumes, the MLflow registry — is reconstructible from what is committed here,
so the repository is the artifact and the running system is a cache of it.

