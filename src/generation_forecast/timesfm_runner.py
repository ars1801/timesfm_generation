"""TimesFM 2.5 forecast wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generation_forecast.cleaning import point_columns
from generation_forecast.metadata import POINTS_BY_ID

CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


def _build_future_index(last_timestamp: pd.Timestamp, horizon_hours: int, step_minutes: int) -> pd.DatetimeIndex:
    if 60 % step_minutes != 0:
        raise ValueError(f"step_minutes must evenly divide 60, got {step_minutes}")

    steps_per_hour = 60 // step_minutes
    periods = horizon_hours * steps_per_hour
    return pd.date_range(
        last_timestamp + pd.Timedelta(minutes=step_minutes),
        periods=periods,
        freq=f"{step_minutes}min",
    )


def _load_timesfm_model(max_context: int, max_horizon: int):
    try:
        import torch
        import timesfm
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM dependencies are not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max(64, max_horizon),
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    return model


def forecast_timesfm(
    cleaned: pd.DataFrame,
    horizon: int,
    context: int = 4096,
    step_minutes: int = 60,
    model=None,
) -> pd.DataFrame:
    """Forecast all generation points with TimesFM 2.5."""
    cleaned = cleaned.sort_values("datetime").set_index("datetime")
    columns = point_columns(cleaned)
    last_timestamp = cleaned.index.max()
    future_index = _build_future_index(last_timestamp, horizon, step_minutes)
    forecast_steps = len(future_index)

    inputs = [
        cleaned[column].tail(context).astype(float).to_numpy(dtype=np.float32)
        for column in columns
    ]
    if model is None:
        model = _load_timesfm_model(max_context=context, max_horizon=forecast_steps)
    point_forecast, quantile_forecast = model.forecast(horizon=forecast_steps, inputs=inputs)

    rows = []
    for series_index, point_id_text in enumerate(columns):
        point = POINTS_BY_ID[int(point_id_text)]
        limit = point.capacity_mw * 1.20
        for step, timestamp in enumerate(future_index):
            raw = float(point_forecast[series_index, step])
            clipped = max(0.0, min(raw, limit))
            row = {
                "datetime": timestamp,
                "point_id": point.point_id,
                "point_name": point.name,
                "zone": point.zone,
                "generation_type": point.generation_type,
                "capacity_mw": point.capacity_mw,
                "model": "timesfm_2p5_200m",
                "forecast_mw": clipped,
                "raw_forecast_mw": raw,
                "forecast_source": CHECKPOINT,
            }
            if quantile_forecast is not None:
                # TimesFM 2.5 returns mean plus q10..q90 in the final dimension.
                row["q10_mw"] = max(0.0, min(float(quantile_forecast[series_index, step, 1]), limit))
                row["q50_mw"] = max(0.0, min(float(quantile_forecast[series_index, step, 5]), limit))
                row["q90_mw"] = max(0.0, min(float(quantile_forecast[series_index, step, 9]), limit))
            rows.append(row)

    return pd.DataFrame(rows)


def load_compiled_timesfm(context: int, horizon: int):
    """Load and compile TimesFM once for repeated forecasts."""
    return _load_timesfm_model(max_context=context, max_horizon=horizon)
