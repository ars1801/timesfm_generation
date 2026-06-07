"""Excel loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from generation_forecast.metadata import EXPECTED_POINT_IDS

DEFAULT_INPUT_DIR = Path("/Users/arsen.serikkaliyev.04icloud.com/Downloads")
FILE_PATTERN = "generation_ti_20*.xlsx"


def normalize_column_name(column: object) -> str:
    text = str(column).strip()
    if text.lower() == "datetime":
        return "datetime"
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def discover_excel_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_PATTERN!r} found in {input_dir}")
    return files


def load_generation_excels(input_dir: Path = DEFAULT_INPUT_DIR) -> pd.DataFrame:
    frames = []
    for file_path in discover_excel_files(input_dir):
        frame = pd.read_excel(file_path, sheet_name=0)
        frame.columns = [normalize_column_name(column) for column in frame.columns]
        if "datetime" not in frame.columns:
            raise ValueError(f"{file_path.name} does not contain a datetime column")
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["source_file"] = file_path.name
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("datetime").reset_index(drop=True)
    return data


def validate_generation_columns(data: pd.DataFrame, freq: str = "h") -> dict[str, object]:
    expected = {str(point_id) for point_id in EXPECTED_POINT_IDS}
    actual = {column for column in data.columns if column not in {"datetime", "source_file"}}
    datetimes = pd.DatetimeIndex(data["datetime"])
    expected_datetimes = pd.date_range(datetimes.min(), datetimes.max(), freq=freq)

    return {
        "rows": int(len(data)),
        "points": int(len(actual)),
        "start": datetimes.min(),
        "end": datetimes.max(),
        "duplicate_hours": int(datetimes.duplicated().sum()),
        "missing_hours": int(len(expected_datetimes.difference(datetimes.drop_duplicates()))),
        "missing_point_ids": sorted(int(point_id) for point_id in expected - actual),
        "extra_point_ids": sorted(int(point_id) for point_id in actual - expected),
    }


def assert_valid_generation_data(data: pd.DataFrame, freq: str = "h") -> dict[str, object]:
    validation = validate_generation_columns(data, freq=freq)
    problems = []
    if validation["missing_point_ids"]:
        problems.append(f"missing point IDs: {validation['missing_point_ids']}")
    if validation["extra_point_ids"]:
        problems.append(f"extra point IDs: {validation['extra_point_ids']}")
    if validation["duplicate_hours"]:
        problems.append(f"duplicate hours: {validation['duplicate_hours']}")
    if validation["missing_hours"]:
        problems.append(f"missing hours: {validation['missing_hours']}")
    if problems:
        raise ValueError("Invalid generation data: " + "; ".join(problems))
    return validation
