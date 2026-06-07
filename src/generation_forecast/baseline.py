"""Simple deterministic baseline forecasts."""

from __future__ import annotations

import pandas as pd

from generation_forecast.cleaning import point_columns
from generation_forecast.metadata import POINTS_BY_ID


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


def forecast_baseline(cleaned: pd.DataFrame, horizon: int, step_minutes: int = 60) -> pd.DataFrame:
    """Forecast with same-hour-yesterday, then same-hour-last-week fallback."""
    cleaned = cleaned.sort_values("datetime").set_index("datetime")
    columns = point_columns(cleaned)
    last_timestamp = cleaned.index.max()
    future_index = _build_future_index(last_timestamp, horizon, step_minutes)

    rows = []
    for point_id_text in columns:
        point = POINTS_BY_ID[int(point_id_text)]
        series = cleaned[point_id_text]
        for timestamp in future_index:
            source_timestamp = timestamp - pd.Timedelta(hours=24)
            fallback_timestamp = timestamp - pd.Timedelta(days=7)
            if source_timestamp in series.index:
                value = series.loc[source_timestamp]
                source = "same_hour_yesterday"
            elif fallback_timestamp in series.index:
                value = series.loc[fallback_timestamp]
                source = "same_hour_last_week"
            else:
                value = series.iloc[-1]
                source = "last_observed"

            limit = point.capacity_mw * 1.20
            rows.append(
                {
                    "datetime": timestamp,
                    "point_id": point.point_id,
                    "point_name": point.name,
                    "zone": point.zone,
                    "generation_type": point.generation_type,
                    "capacity_mw": point.capacity_mw,
                    "model": "baseline_same_hour_yesterday",
                    "forecast_mw": max(0.0, min(float(value), limit)),
                    "raw_forecast_mw": float(value),
                    "forecast_source": source,
                }
            )

    return pd.DataFrame(rows)
