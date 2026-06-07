"""Command-line interface for the generation forecast MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from generation_forecast.baseline import forecast_baseline
from generation_forecast.clickhouse import get_clickhouse_client
from generation_forecast.clickhouse_generation import (
    load_generation_history_from_clickhouse,
    resolve_latest_complete_timestamp,
    sync_generation_forecasts_to_clickhouse,
)
from generation_forecast.cleaning import clean_generation_data, to_long_frame
from generation_forecast.data import assert_valid_generation_data
from generation_forecast.metrics import score_forecast, score_merged_forecast
from generation_forecast.timesfm_runner import forecast_timesfm, load_compiled_timesfm

DEFAULT_CLICKHOUSE_URL = None
DEFAULT_CLICKHOUSE_DATABASE = None
DEFAULT_CLICKHOUSE_USERNAME = None
DEFAULT_CLICKHOUSE_PASSWORD = None
DEFAULT_CLICKHOUSE_TIMEOUT = None


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--horizon", type=int, default=36)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--step-minutes", type=int, default=None)
    parser.add_argument("--model", choices=("baseline", "timesfm", "both"), default="baseline")
    parser.add_argument("--persist-clickhouse", action="store_true")
    parser.add_argument("--clickhouse-url", default=DEFAULT_CLICKHOUSE_URL)
    parser.add_argument("--clickhouse-database", default=DEFAULT_CLICKHOUSE_DATABASE)
    parser.add_argument("--clickhouse-username", default=DEFAULT_CLICKHOUSE_USERNAME)
    parser.add_argument("--clickhouse-password", default=DEFAULT_CLICKHOUSE_PASSWORD)
    parser.add_argument("--clickhouse-timeout", type=int, default=DEFAULT_CLICKHOUSE_TIMEOUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast generation points from ClickHouse aggregates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("forecast", "backtest"):
        subparser = subparsers.add_parser(command)
        _add_common_arguments(subparser)

    rolling = subparsers.add_parser("rolling-backtest")
    _add_common_arguments(rolling)
    rolling.add_argument("--freq", default="7D")

    return parser.parse_args()


def _step_minutes(args: argparse.Namespace) -> int:
    return args.step_minutes if args.step_minutes is not None else 10


def _build_clickhouse_client(args: argparse.Namespace):
    return get_clickhouse_client(
        url=args.clickhouse_url,
        database=args.clickhouse_database,
        username=args.clickhouse_username,
        password=args.clickhouse_password,
        timeout=args.clickhouse_timeout,
    )


def _load_clickhouse_data(
    *,
    client,
    output_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_generation_history_from_clickhouse(client, start, end, step_minutes=step_minutes)
    assert_valid_generation_data(raw, freq=f"{step_minutes}min")
    cleaned, report = clean_generation_data(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "cleaning_report.csv", index=False)
    return cleaned, report


def _default_history_start(end: pd.Timestamp, step_minutes: int, context: int) -> pd.Timestamp:
    padding_steps = max(context + (24 * 60 // step_minutes), context)
    return end - pd.Timedelta(minutes=step_minutes * padding_steps)


def _load_data_for_forecast(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
    step_minutes = _step_minutes(args)
    client = _build_clickhouse_client(args)
    latest_complete = resolve_latest_complete_timestamp(client)
    history_start = _default_history_start(latest_complete, step_minutes, args.context)
    cleaned, report = _load_clickhouse_data(
        client=client,
        output_dir=args.output_dir,
        start=history_start,
        end=latest_complete,
        step_minutes=step_minutes,
    )
    return cleaned, report, client


def _load_data_for_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
    step_minutes = _step_minutes(args)
    client = _build_clickhouse_client(args)
    latest_complete = resolve_latest_complete_timestamp(client)
    cutoff = latest_complete - pd.Timedelta(hours=args.horizon)
    history_start = _default_history_start(cutoff, step_minutes, args.context)
    cleaned, report = _load_clickhouse_data(
        client=client,
        output_dir=args.output_dir,
        start=history_start,
        end=latest_complete,
        step_minutes=step_minutes,
    )
    return cleaned, report, client


def _load_data_for_rolling_backtest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
    step_minutes = _step_minutes(args)
    client = _build_clickhouse_client(args)
    latest_complete = resolve_latest_complete_timestamp(client)
    cleaned, report = _load_clickhouse_data(
        client=client,
        output_dir=args.output_dir,
        start=latest_complete - pd.Timedelta(minutes=step_minutes * args.context),
        end=latest_complete,
        step_minutes=step_minutes,
    )
    return cleaned, report, client


def run_model(
    model_name: str,
    cleaned: pd.DataFrame,
    horizon: int,
    context: int,
    *,
    step_minutes: int,
) -> pd.DataFrame:
    if model_name == "baseline":
        return forecast_baseline(cleaned, horizon=horizon, step_minutes=step_minutes)
    if model_name == "timesfm":
        return forecast_timesfm(cleaned, horizon=horizon, context=context, step_minutes=step_minutes)
    if model_name == "both":
        return pd.concat(
            [
                forecast_baseline(cleaned, horizon=horizon, step_minutes=step_minutes),
                forecast_timesfm(cleaned, horizon=horizon, context=context, step_minutes=step_minutes),
            ],
            ignore_index=True,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def run_model_with_loaded_timesfm(
    model_name: str,
    cleaned: pd.DataFrame,
    horizon: int,
    context: int,
    *,
    step_minutes: int,
    timesfm_model=None,
) -> pd.DataFrame:
    if model_name == "baseline":
        return forecast_baseline(cleaned, horizon=horizon, step_minutes=step_minutes)
    if model_name == "timesfm":
        return forecast_timesfm(
            cleaned,
            horizon=horizon,
            context=context,
            step_minutes=step_minutes,
            model=timesfm_model,
        )
    raise ValueError(f"Unsupported rolling model: {model_name}")


def write_forecast_outputs(forecast: pd.DataFrame, output_dir: Path, horizon: int, model_name: str) -> None:
    suffix = f"{horizon}h_latest_{model_name}"
    forecast_path = output_dir / f"forecast_{suffix}.csv"
    wide_path = output_dir / f"forecast_{suffix}_wide.csv"

    forecast.to_csv(forecast_path, index=False)
    wide = forecast.pivot_table(
        index="datetime",
        columns=["model", "point_id"],
        values="forecast_mw",
        aggfunc="first",
    )
    wide.to_csv(wide_path)

    total = (
        forecast.groupby(["datetime", "model"], as_index=False)["forecast_mw"]
        .sum()
        .rename(columns={"forecast_mw": "total_forecast_mw"})
    )
    total.to_csv(output_dir / f"forecast_{suffix}_total.csv", index=False)

    print(f"Wrote {forecast_path}")
    print(f"Wrote {wide_path}")


def command_forecast(args: argparse.Namespace) -> None:
    step_minutes = _step_minutes(args)
    cleaned, report, client = _load_data_for_forecast(args)
    if args.persist_clickhouse and args.model == "both":
        raise ValueError("Cannot persist combined forecasts to ClickHouse because generation_forecasts has no model column")
    forecast = run_model(args.model, cleaned, args.horizon, args.context, step_minutes=step_minutes)
    write_forecast_outputs(forecast, args.output_dir, args.horizon, args.model)
    if args.persist_clickhouse:
        inserted = sync_generation_forecasts_to_clickhouse(
            client,
            forecast,
        )
        print(f"Synced {inserted} forecast rows to ClickHouse")
    print(
        "Cleaning summary: "
        f"{report.attrs.get('total_invalid_cells', 'n/a')} invalid cells, "
        f"{report.attrs.get('all_zero_rows', 'n/a')} all-zero rows"
    )


def command_backtest(args: argparse.Namespace) -> None:
    step_minutes = _step_minutes(args)
    cleaned, _, _ = _load_data_for_backtest(args)
    client = _build_clickhouse_client(args)
    latest_complete = resolve_latest_complete_timestamp(client)
    cutoff = latest_complete - pd.Timedelta(hours=args.horizon)
    history = cleaned.loc[cleaned["datetime"] <= cutoff].copy()
    actual = to_long_frame(cleaned.loc[cleaned["datetime"] > cutoff].copy())

    forecast = run_model(args.model, history, args.horizon, args.context, step_minutes=step_minutes)

    overall, by_point = score_forecast(forecast, actual)
    suffix = f"{args.horizon}h_backtest_{args.model}"
    forecast.to_csv(args.output_dir / f"forecast_{suffix}.csv", index=False)
    overall.to_csv(args.output_dir / f"metrics_{suffix}_overall.csv", index=False)
    by_point.to_csv(args.output_dir / f"metrics_{suffix}_by_point.csv", index=False)

    print(overall.to_string(index=False))


def command_rolling_backtest(args: argparse.Namespace) -> None:
    step_minutes = _step_minutes(args)
    cleaned, _, _ = _load_data_for_rolling_backtest(args)
    cleaned = cleaned.sort_values("datetime").reset_index(drop=True)
    first_datetime = cleaned["datetime"].min()
    last_datetime = cleaned["datetime"].max()
    start = first_datetime + pd.Timedelta(minutes=step_minutes * args.context)
    latest_origin = last_datetime - pd.Timedelta(hours=args.horizon)
    if start > latest_origin:
        raise ValueError(f"Rolling backtest window is empty: start={start}, end={latest_origin}")

    origins = pd.date_range(start=start, end=latest_origin, freq=args.freq)
    if len(origins) == 0:
        raise ValueError(f"No origins generated for start={start}, end={latest_origin}, freq={args.freq}")

    model_names = ["baseline", "timesfm"] if args.model == "both" else [args.model]
    timesfm_model = None
    if "timesfm" in model_names:
        timesfm_model = load_compiled_timesfm(context=args.context, horizon=int(args.horizon * 60 // step_minutes))

    actual_long = to_long_frame(cleaned)
    merged_parts = []
    forecast_parts = []
    for index, origin in enumerate(origins, start=1):
        history = cleaned.loc[cleaned["datetime"] <= origin].copy()
        if len(history) < max(args.context // 4, 24):
            continue
        for model_name in model_names:
            forecast = run_model_with_loaded_timesfm(
                model_name,
                history,
                args.horizon,
                args.context,
                step_minutes=step_minutes,
                timesfm_model=timesfm_model,
            )
            forecast["origin_datetime"] = origin
            forecast["horizon_step"] = (
                (forecast["datetime"] - origin) / pd.Timedelta(minutes=step_minutes)
            ).astype(int)
            forecast_parts.append(forecast)
            merged = forecast.merge(actual_long, on=["datetime", "point_id"], how="inner")
            merged_parts.append(merged)
        print(f"Finished origin {index}/{len(origins)}: {origin}", flush=True)

    if not merged_parts:
        raise ValueError("Rolling backtest produced no scored forecasts")

    forecasts = pd.concat(forecast_parts, ignore_index=True)
    merged_all = pd.concat(merged_parts, ignore_index=True)
    overall, by_point, by_origin = score_merged_forecast(merged_all)

    suffix = f"{args.horizon}h_rolling_{args.model}_{args.freq.replace(' ', '')}"
    forecasts.to_csv(args.output_dir / f"forecast_{suffix}.csv", index=False)
    merged_all.to_csv(args.output_dir / f"scored_{suffix}.csv", index=False)
    overall.to_csv(args.output_dir / f"metrics_{suffix}_overall.csv", index=False)
    by_point.to_csv(args.output_dir / f"metrics_{suffix}_by_point.csv", index=False)
    by_origin.to_csv(args.output_dir / f"metrics_{suffix}_by_origin.csv", index=False)
    print(overall.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.command == "forecast":
        command_forecast(args)
    elif args.command == "backtest":
        command_backtest(args)
    elif args.command == "rolling-backtest":
        command_rolling_backtest(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
