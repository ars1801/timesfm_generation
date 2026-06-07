"""Cleaning rules for generation time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from generation_forecast.metadata import EXPECTED_POINT_IDS, POINTS_BY_ID

SENTINEL_VALUE = 2_147_483_647
ABSURD_MW_LIMIT = 10_000
CAPACITY_MULTIPLIER_LIMIT = 1.20


def point_columns(data: pd.DataFrame) -> list[str]:
    return [str(point_id) for point_id in EXPECTED_POINT_IDS if str(point_id) in data.columns]


def clean_generation_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = data.copy()
    columns = point_columns(cleaned)
    values = cleaned[columns].apply(pd.to_numeric, errors="coerce")

    capacity = pd.Series(
        {str(point_id): POINTS_BY_ID[point_id].capacity_mw for point_id in EXPECTED_POINT_IDS}
    )[columns]

    all_zero_row = values.fillna(0).sum(axis=1).eq(0)
    invalid = values.isna()
    invalid |= values.lt(0)
    invalid |= values.eq(SENTINEL_VALUE)
    invalid |= values.gt(ABSURD_MW_LIMIT)
    invalid |= values.gt(capacity * CAPACITY_MULTIPLIER_LIMIT, axis=1)
    invalid.loc[all_zero_row, :] = True

    repaired = values.mask(invalid)
    repaired = repaired.interpolate(method="linear", limit_direction="both")
    repaired = repaired.fillna(0)
    repaired = repaired.clip(lower=0, upper=capacity * CAPACITY_MULTIPLIER_LIMIT, axis=1)

    cleaned[columns] = repaired

    report_rows = []
    for column in columns:
        point = POINTS_BY_ID[int(column)]
        mask = invalid[column]
        clean_series = repaired[column]
        report_rows.append(
            {
                "point_id": point.point_id,
                "point_name": point.name,
                "capacity_mw": point.capacity_mw,
                "invalid_cells": int(mask.sum()),
                "invalid_pct": float(mask.mean() * 100),
                "raw_missing_cells": int(values[column].isna().sum()),
                "zero_pct_after_cleaning": float(clean_series.eq(0).mean() * 100),
                "max_mw_after_cleaning": float(clean_series.max()),
                "mean_mw_after_cleaning": float(clean_series.mean()),
            }
        )

    report = pd.DataFrame(report_rows).sort_values(
        ["invalid_cells", "point_id"], ascending=[False, True]
    )
    report.attrs["all_zero_rows"] = int(all_zero_row.sum())
    report.attrs["total_invalid_cells"] = int(invalid.sum().sum())
    return cleaned, report


def to_long_frame(data: pd.DataFrame) -> pd.DataFrame:
    columns = point_columns(data)
    long = data[["datetime", *columns]].melt(
        id_vars="datetime", var_name="point_id", value_name="generation_mw"
    )
    long["point_id"] = long["point_id"].astype(int)
    return long
