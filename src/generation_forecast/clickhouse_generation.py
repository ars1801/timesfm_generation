"""ClickHouse-backed generation history and forecast persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from generation_forecast.metadata import EXPECTED_POINT_IDS

GENERATION_TZ = ZoneInfo("Etc/GMT-5")
GENERATION_TZ_NAME = "Etc/GMT-5"


def _to_generation_tz(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(GENERATION_TZ)
    return timestamp.tz_convert(GENERATION_TZ)


def _clickhouse_dt_literal(value: datetime | pd.Timestamp) -> str:
    return _to_generation_tz(value).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_timestamp(value: str | datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(GENERATION_TZ).tz_localize(None)
    return timestamp.tz_convert(GENERATION_TZ).tz_localize(None)


def resolve_latest_complete_timestamp(client, expected_point_count: int | None = None) -> pd.Timestamp:
    expected_point_count = expected_point_count or len(EXPECTED_POINT_IDS)
    rows = client.query_json_each_row(
        f"""
        SELECT timestamp
        FROM
        (
            SELECT
                timestamp,
                countDistinct(point_id) AS points
            FROM generation_aggregated
            GROUP BY timestamp
        )
        WHERE points >= {expected_point_count}
        ORDER BY timestamp DESC
        LIMIT 1
        """
    )
    if not rows:
        raise ValueError("Could not resolve the latest complete generation timestamp from ClickHouse")
    return pd.Timestamp(rows[0]["timestamp"])


def load_generation_history_from_clickhouse(
    client,
    start: datetime | pd.Timestamp,
    end: datetime | pd.Timestamp,
    *,
    step_minutes: int,
) -> pd.DataFrame:
    """Load point-level generation aggregates from ClickHouse as a wide frame."""
    columns = [str(point_id) for point_id in EXPECTED_POINT_IDS]
    rows = client.query_json_each_row(
        f"""
        SELECT
            timestamp,
            point_id,
            generation_avg_mw AS generation_mw
        FROM generation_aggregated
        WHERE point_id IN ({", ".join(repr(column) for column in columns)})
          AND timestamp >= toDateTime('{_clickhouse_dt_literal(start)}', '{GENERATION_TZ_NAME}')
          AND timestamp <= toDateTime('{_clickhouse_dt_literal(end)}', '{GENERATION_TZ_NAME}')
        ORDER BY timestamp, point_id
        """
    )

    if not rows:
        raise ValueError("No generation rows returned from ClickHouse for the requested window")

    frame = pd.DataFrame(rows)
    observed_points = {str(point_id) for point_id in frame["point_id"].unique()}
    missing_points = [int(point_id) for point_id in columns if point_id not in observed_points]
    if missing_points:
        raise ValueError(f"Missing generation points in ClickHouse response: {missing_points}")
    frame["datetime"] = frame["timestamp"].map(_normalize_timestamp)
    frame["point_id"] = frame["point_id"].astype(str)
    frame["generation_mw"] = pd.to_numeric(frame["generation_mw"], errors="coerce")

    wide = frame.pivot_table(index="datetime", columns="point_id", values="generation_mw", aggfunc="last")
    wide = wide.reindex(columns=columns)

    start_ts = _normalize_timestamp(start)
    end_ts = _normalize_timestamp(end)
    full_index = pd.date_range(start_ts, end_ts, freq=f"{step_minutes}min")
    wide = wide.reindex(full_index)
    wide.index.name = "datetime"
    wide = wide.reset_index()
    wide["source_file"] = "clickhouse:generation_aggregated"
    return wide[["datetime", *columns, "source_file"]]


def sync_generation_forecasts_to_clickhouse(
    client,
    forecast: pd.DataFrame,
    *,
    created_at: datetime | None = None,
) -> int:
    """Persist point-level generation forecasts to generation_forecasts."""
    created_at = created_at or datetime.now(timezone.utc)
    rows = []
    for _, row in forecast.iterrows():
        rows.append(
            {
                "forecast_created_at": _to_generation_tz(created_at).strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": _to_generation_tz(row["datetime"]).strftime("%Y-%m-%d %H:%M:%S"),
                "point_id": str(row["point_id"]),
                "expected_generation_mw": float(row["forecast_mw"]),
            }
        )

    client.insert_json_each_row("generation_forecasts", rows)
    return len(rows)
