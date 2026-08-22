# Exogenous and calendar features

Status: approved 2026-08-22. Supersedes the four-channel `ml_features` layout.

## Why

The champions are measurably weak at long horizons. Per-horizon temperature RMSE over
the 113,856-window test block, measured 2026-08-22:

| horizon | FastLSTM@v6 | Transformer@v3 | seasonal-naive |
|---|---|---|---|
| t+1 | 0.874 °C | 0.913 | 2.524 |
| t+6 | 1.597 | 1.578 | 2.524 |
| t+12 | 1.903 | 1.924 | 2.524 |
| t+24 | **2.303** | 2.305 | 2.524 |
| mean | 1.873 | 1.881 | 2.524 |

Two things follow. The models beat seasonal-naive by 65% at t+1 and by only **9%** at
t+24 — by the end of the horizon they are barely better than "the same hour yesterday".
And two quite different architectures land within 0.4% of each other and swap places at
t+6, so architecture is not where the headroom is.

The models are given four raw channels and nothing else. They have no clock, no
calendar, and no synoptic state — nothing that says what time it will be at t+24, and
nothing about the pressure, cloud or air mass that determines what happens next. This
adds all three.

## Goals

- Give the models the information a 24-hour forecast physically depends on.
- Make per-horizon error visible, in `evaluate_and_promote` and in the serving API, so
  this change and every later one can be judged where it matters.
- Handle a champion that predates a feature-schema change without crashing the gate.

## Non-goals

- **Architecture and hyperparameters stay fixed.** Changing capacity at the same time
  as the inputs would make the result unattributable. If 16 channels turn out to need
  more capacity, that is a separate experiment with its own verdict.
- **No known-future covariate input.** Feeding the model the *future* calendar as a
  second input would be stronger, but it means a two-argument `forward()`, a
  two-input ONNX graph and future-calendar construction in `batch_inference`. The
  72-hour window covers three full diurnal cycles, so sin/cos extrapolation is
  learnable, and the seasonal signal is near-constant across a window anyway. Revisit
  only if the horizon curve barely moves.
- No multi-site work. Still one fixed coordinate.

## The feature vector

`ml_features.features` becomes a `list<float32>` of **16** values. The first four keep
their identity and position, which is what lets `batch_inference` and the serving
layer's `denormalize(x[0])` stay untouched.

| # | Channel | Source column | Transform |
|---|---|---|---|
| 0 | `temperature` | `temperature_c` | standardize |
| 1 | `humidity` | `humidity_percent` | standardize |
| 2 | `precipitation` | `precipitation_mm` | standardize |
| 3 | `wind_speed` | `wind_speed_kmh` | standardize |
| 4 | `pressure` | `pressure_msl_hpa` | standardize |
| 5 | `dew_point` | `dew_point_c` | standardize |
| 6 | `cloud_cover` | `cloud_cover_percent` | standardize |
| 7 | `shortwave_radiation` | `shortwave_radiation_wm2` | standardize |
| 8 | `soil_temperature` | `soil_temperature_c` | standardize |
| 9 | `soil_moisture` | `soil_moisture_m3m3` | standardize |
| 10 | `wind_dir_sin` | `wind_direction_deg` | `sin(2π·deg/360)` |
| 11 | `wind_dir_cos` | `wind_direction_deg` | `cos(2π·deg/360)` |
| 12 | `hour_sin` | `timestamp` | `sin(2π·hour/24)` |
| 13 | `hour_cos` | `timestamp` | `cos(2π·hour/24)` |
| 14 | `doy_sin` | `timestamp` | `sin(2π·doy/365.25)` |
| 15 | `doy_cos` | `timestamp` | `cos(2π·doy/365.25)` |

Channels 10-15 are already bounded in [-1, 1] and are **not** standardized. They are
still written to `weather.scaling_parameters` with `mean=0.0, std=1.0`, so that table
remains a complete description of the vector rather than a description of part of it.

**Model output stays 4 channels** — the original weather variables. The calendar is
deterministic and forecasting the exogenous inputs would only make the task harder.

### Why these and not the others

Each added channel has a distinct physical mechanism for 24-hour temperature:

- **`pressure_msl`** — synoptic state. Pressure tendency is the classic indicator that
  an air mass is changing, which is precisely the t+24 question.
- **`dew_point_2m`** — absolute moisture. Temperature rarely falls below the dew point,
  so this bounds the overnight minimum.
- **`cloud_cover`** — modulates daytime heating and night-time radiative cooling.
- **`shortwave_radiation`** — actual solar forcing. Together with the calendar channels
  it separates "clear" from "cloudy" quantitatively rather than by proxy.
- **`soil_temperature_0_to_7cm`** — the surface's thermal memory, a slow state variable
  carrying multi-day history that the air temperature alone does not.
- **`soil_moisture_0_to_7cm`** — sets the latent/sensible heat split, which governs
  diurnal amplitude.
- **`wind_direction_10m`** — advection. At 40.98 N / 27.51 E southerly flow is warm and
  northerly is cold. Circular, so sin/cos rather than degrees, or 359° and 1° would sit
  at opposite ends of the range.

Rejected: `apparent_temperature`, `et0_fao_evapotranspiration` and
`vapour_pressure_deficit` are computed from channels we already carry.
`surface_pressure` differs from `pressure_msl` by a constant at a fixed altitude.
`sunshine_duration` is near-collinear with `shortwave_radiation`. `wind_gusts_10m` is
largely collinear with wind speed and has the weakest independent mechanism.
`snowfall`/`snow_depth` are near-zero at this site, so there is no training signal.
Channels are not free: with `d_model=32` and `hidden_dim=64` the capacity is small and
redundant inputs cost more than they carry.

### Availability

Probed against the archive API for 1940-01, 1975-06 and 2026-08. Every variable above
is present in all three. The only nulls are the first 7 hours of 1940-01-01, in
`precipitation`, `shortwave_radiation` and `wind_gusts_10m` — flux variables that need
a preceding accumulation interval. `precipitation` already has this gap, which is why
the current table starts at 1940-01-01 07:00. **No history is lost.**

## Component changes

### `jobs/weather_etl/weather_etl.py`

Fetch the seven new variables. Extend `run_data_quality_checks` with physical bounds:

| Column | Bound |
|---|---|
| `pressure_msl_hpa` | 870 – 1085 |
| `dew_point_c` | −60 – 60 |
| `cloud_cover_percent` | 0 – 100 |
| `shortwave_radiation_wm2` | 0 – 1400 |
| `soil_temperature_c` | −60 – 70 |
| `soil_moisture_m3m3` | 0 – 1 |
| `wind_direction_deg` | 0 – 360 |

Add one cross-column invariant: **dew point must not exceed temperature** (allowing
0.5 °C for ERA5's numerical noise). Air cannot hold more moisture than saturation, so a
violation means the two columns have been misaligned — exactly the failure a bounds
check on either column alone cannot see.

### `jobs/feature_engineering/feature_engineering.py`

`FEATURE_COLS` grows to the ten standardized columns. Two new lists describe the
channels that bypass standardization: the wind-direction pair, derived from
`wind_direction_deg`, and the four calendar channels, derived from `timestamp` with
`F.hour` and `F.dayofyear`. `standardize()` emits all sixteen in the order above.
`write_scaling_parameters` writes sixteen rows, with `(0.0, 1.0)` for channels 10-15.

`compute_global_stats` and `stats_drifted` operate only on the ten standardized
columns — a cyclical channel has no drift to detect. `rebuild_reason`'s row-count
invariant is unaffected, being independent of the column set.

One trap: `load_published_stats` currently reconstructs its mapping by zipping
`FEATURE_COLS` against `SERVING_FEATURE_NAMES`. Those lists stop being the same length
here — sixteen channel names, ten standardized source columns — so the lookup must
zip against the standardized subset explicitly rather than against the full name list,
or an incremental run silently normalizes with the wrong parameters.

### `jobs/model_training/data_loader.py`

`OUTPUT_CHANNELS = 4`. `__getitem__` returns `y` sliced to the first `OUTPUT_CHANNELS`
channels; `x` keeps all sixteen.

### `jobs/model_training/models.py`

`input_dim` and `output_dim` become genuinely independent. Both constructors already
take them separately; the call sites in `train_lstm.py` and `train_transformer.py`
currently pass `feature_dim` for both and must pass `input_dim=16, output_dim=4`.

### `jobs/model_training/baselines.py`

Both baselines currently return every input channel. They must return only the output
channels, or they cannot be compared against `y` at all.

### `jobs/model_training/evaluate_and_promote.py`

**Per-horizon metrics.** `Scores` gains `horizon_rmse`, one RMSE per forecast hour for
the temperature channel, converted to °C by the published temperature standard
deviation read from `weather.scaling_parameters`. All 24 are logged to MLflow per
predictor as `{predictor}_temp_rmse_c_h01..h24`; the task log prints t+1, t+6, t+12 and
t+24 plus the mean, which is the shape of the table at the top of this document.

**Incompatible champion.** Before the benchmark, run one batch through the champion. If
it raises, that alone is not proof the schema moved on: a CUDA OOM, a flaky kernel or a
genuine bug in an otherwise-compatible champion raises from the exact same forward
pass. So the same batch is also run through the challenger, which was just trained
against the current vector and is therefore compatible by construction
(`champion_alone_cannot_score`, `evaluate_and_promote.py:122-142`). Only when the
champion raises and the challenger succeeds is it treated as a schema incompatibility:
log that loudly, promote the challenger as a fresh baseline, move the old champion to a
`retired_champion` alias, and record `promotion_decision=PROMOTED_SCHEMA_CHANGE`. Do
not set `previous_champion`: that alias implies a model that could be rolled back to,
and this one cannot run at all. If neither model can be scored, the task raises instead
of promoting anything — that pattern points at a broken environment, not a schema
change, and nothing should be crowned champion on the strength of a guess.

This path is defensive for the migration below — the registry is being wiped, so it
will not fire — but a feature-schema change will happen again, and today it would
crash the gate rather than report anything.

### `jobs/model_training/batch_inference.py` and `jobs/serving/api.py`

`forecast_predictions` gains a required `horizon` field, 1-24, written by
`batch_inference` from the position of each predicted hour. `/api/v1/metrics/residuals`
groups by `(model_name, model_version, horizon)` so the error-versus-horizon curve is
visible from the serving side too.

The table is dropped and recreated by the migration, so the field can be required and
no backfill is needed.

## Migration

Destructive and deliberate. All of it is reproducible from Open-Meteo, and
`/mnt/c/lakehouse-backup-2026-08-22/` holds the pre-change state.

1. Drop `weather.observations`, `weather.ml_features`, `weather.scaling_parameters`
   and `weather.forecast_predictions`.
2. Delete the `Weather_Forecaster_FastLSTM` and `Weather_Forecaster_Transformer`
   registered models. Their inputs no longer exist; they cannot serve or be scored.
3. Rebuild the job images.
4. Run `01` (full backfill from 1940, ~11 minutes), which triggers `02` (full rebuild)
   and `04`.
5. Trigger `03a` and `03b`. Both take the SCRATCH path against an empty registry and
   auto-promote as initial baselines.
6. Re-run `04` to produce forecasts from the new champions.

## Testing

Every suite runs in the image that carries its dependencies, as today.

| Suite | New coverage |
|---|---|
| `test_weather_etl.py` | new bounds; the dew-point-above-temperature invariant fails |
| `test_feature_engineering.py` | cyclical encodings are continuous across the 359°→0° and Dec-31→Jan-1 seams; cyclical channels are written with identity scaling; the scaling table describes all 16 channels |
| `test_data_loader.py` | `y` carries only the output channels while `x` carries all inputs |
| `test_baselines.py` | both baselines return output-channel width from a 16-channel context |
| `test_evaluation.py` | per-horizon RMSE matches hand-computable errors; an incompatible champion is promoted-around rather than raising |
| `test_serving.py` | residuals group per horizon |

## How the result gets judged

The new models are the first registrations against an empty registry, so they promote
automatically and the gate's verdict measures nothing this round. The measurement is
the per-horizon table, compared directly against the numbers at the top of this
document. The baselines are unchanged by any of this (2.524 °C flat), so they remain a
fixed reference across the schema change.

The honest possible outcomes are that long-horizon error drops materially, that it
barely moves — in which case the next experiment is capacity or known-future
covariates — or that it worsens because 16 channels overwhelm a small model. All three
are informative, and the per-horizon table is what distinguishes them.
