# Generation Forecast MVP

Forecasting generation points in Kazakhstan with Google TimesFM 2.5.

The workflow reads point-level generation history from ClickHouse
`generation_aggregated`, forecasts the next 36 hours at 10-minute resolution,
and can write the result back to `generation_forecasts`.

## Data Model

The forecast pipeline expects these ClickHouse tables:

- `generation_points`
- `generation_aggregated`
- `generation_forecasts`

Zone totals are available through `zone_generation_forecasts_view` in the main
backend project.

## Environment

Default ClickHouse connection values:

- URL: `http://localhost:8123`
- Database: `kegoc`
- Username: `default`
- Password: empty

Override them with CLI flags:

- `--clickhouse-url`
- `--clickhouse-database`
- `--clickhouse-username`
- `--clickhouse-password`
- `--clickhouse-timeout`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Forecast

```bash
PYTHONPATH=src .venv/bin/python -m generation_forecast.cli forecast \
  --model timesfm \
  --horizon 36 \
  --step-minutes 10 \
  --output-dir outputs \
  --persist-clickhouse
```

Outputs:

- `outputs/forecast_36h_latest_timesfm.csv`
- `outputs/forecast_36h_latest_timesfm_wide.csv`
- `outputs/forecast_36h_latest_timesfm_total.csv`
- `outputs/cleaning_report.csv`

If `--persist-clickhouse` is set, the forecast rows are inserted into
`generation_forecasts`.

## Baseline

```bash
PYTHONPATH=src .venv/bin/python -m generation_forecast.cli forecast \
  --model baseline \
  --horizon 36 \
  --step-minutes 10 \
  --output-dir outputs
```

## Backtest

```bash
PYTHONPATH=src .venv/bin/python -m generation_forecast.cli backtest \
  --model timesfm \
  --horizon 36 \
  --step-minutes 10 \
  --output-dir outputs
```

## Rolling Backtest

```bash
PYTHONPATH=src .venv/bin/python -m generation_forecast.cli rolling-backtest \
  --model timesfm \
  --horizon 36 \
  --step-minutes 10 \
  --freq 30D \
  --output-dir outputs
```

## Notes

- `horizon 36` means 36 hours.
- With `--step-minutes 10`, the model predicts 216 steps.
- The cleaning step removes missing, negative, sentinel, and out-of-range
  values before forecasting.
