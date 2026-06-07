"""Backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_forecast(forecast: pd.DataFrame, actual_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = forecast.merge(
        actual_long,
        on=["datetime", "point_id"],
        how="inner",
    )
    merged["error_mw"] = merged["forecast_mw"] - merged["generation_mw"]
    merged["abs_error_mw"] = merged["error_mw"].abs()
    merged["squared_error_mw"] = merged["error_mw"] ** 2
    denom = (merged["forecast_mw"].abs() + merged["generation_mw"].abs()).replace(0, np.nan)
    merged["smape"] = (2 * merged["abs_error_mw"] / denom).fillna(0)
    merged["ape"] = np.where(
        merged["generation_mw"].abs() > 0,
        merged["abs_error_mw"] / merged["generation_mw"].abs(),
        np.nan,
    )

    by_point = (
        merged.groupby(["point_id", "point_name", "model"], as_index=False)
        .agg(
            observations=("generation_mw", "size"),
            mae_mw=("abs_error_mw", "mean"),
            rmse_mw=("squared_error_mw", lambda value: float(np.sqrt(value.mean()))),
            mape=("ape", "mean"),
            smape=("smape", "mean"),
        )
        .sort_values("mae_mw", ascending=False)
    )

    actual_sum = merged["generation_mw"].abs().sum()
    overall = pd.DataFrame(
        [
            {
                "model": merged["model"].iloc[0] if not merged.empty else "unknown",
                "observations": int(len(merged)),
                "mae_mw": float(merged["abs_error_mw"].mean()) if not merged.empty else np.nan,
                "rmse_mw": float(np.sqrt(merged["squared_error_mw"].mean())) if not merged.empty else np.nan,
                "mape": float(merged["ape"].mean()) if not merged.empty else np.nan,
                "wmape": float(merged["abs_error_mw"].sum() / actual_sum) if actual_sum else np.nan,
                "smape": float(merged["smape"].mean()) if not merged.empty else np.nan,
            }
        ]
    )
    return overall, by_point


def score_merged_forecast(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score a pre-merged forecast frame, preserving rolling-origin metadata."""
    scored = merged.copy()
    scored["error_mw"] = scored["forecast_mw"] - scored["generation_mw"]
    scored["abs_error_mw"] = scored["error_mw"].abs()
    scored["squared_error_mw"] = scored["error_mw"] ** 2
    denom = (scored["forecast_mw"].abs() + scored["generation_mw"].abs()).replace(0, np.nan)
    scored["smape"] = (2 * scored["abs_error_mw"] / denom).fillna(0)
    scored["ape"] = np.where(
        scored["generation_mw"].abs() > 0,
        scored["abs_error_mw"] / scored["generation_mw"].abs(),
        np.nan,
    )

    def wmape(group: pd.DataFrame) -> float:
        actual_sum = group["generation_mw"].abs().sum()
        if actual_sum == 0:
            return np.nan
        return float(group["abs_error_mw"].sum() / actual_sum)

    overall_rows = []
    for model, group in scored.groupby("model"):
        overall_rows.append(
            {
                "model": model,
                "origins": int(group["origin_datetime"].nunique()) if "origin_datetime" in group else 1,
                "observations": int(len(group)),
                "mae_mw": float(group["abs_error_mw"].mean()),
                "rmse_mw": float(np.sqrt(group["squared_error_mw"].mean())),
                "mape": float(group["ape"].mean()),
                "wmape": wmape(group),
                "smape": float(group["smape"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows).sort_values("mae_mw")

    by_point = (
        scored.groupby(["point_id", "point_name", "model"], as_index=False)
        .agg(
            observations=("generation_mw", "size"),
            mae_mw=("abs_error_mw", "mean"),
            rmse_mw=("squared_error_mw", lambda value: float(np.sqrt(value.mean()))),
            mape=("ape", "mean"),
            smape=("smape", "mean"),
        )
        .sort_values("mae_mw", ascending=False)
    )
    point_wmape = (
        scored.groupby(["point_id", "model"])
        .apply(wmape, include_groups=False)
        .rename("wmape")
        .reset_index()
    )
    by_point = by_point.merge(point_wmape, on=["point_id", "model"], how="left")

    by_origin = (
        scored.groupby(["origin_datetime", "model"], as_index=False)
        .agg(
            observations=("generation_mw", "size"),
            mae_mw=("abs_error_mw", "mean"),
            rmse_mw=("squared_error_mw", lambda value: float(np.sqrt(value.mean()))),
            mape=("ape", "mean"),
            smape=("smape", "mean"),
        )
        .sort_values("origin_datetime")
    )
    origin_wmape = (
        scored.groupby(["origin_datetime", "model"])
        .apply(wmape, include_groups=False)
        .rename("wmape")
        .reset_index()
    )
    by_origin = by_origin.merge(origin_wmape, on=["origin_datetime", "model"], how="left")
    return overall, by_point, by_origin
