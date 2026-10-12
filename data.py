"""Stage 1: validated, timezone-aware daily durations from Volunteer alert intervals.

This module describes *recorded* alerts, not confirmed hazards or live alerts.
It does not train a forecast or prove availability of historical records in real time.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

KYIV = ZoneInfo("Europe/Kyiv")
UTC = timezone.utc
REQUIRED_COLUMNS = {"region", "started_at", "finished_at", "naive"}

# Known categories in the supplied dataset. Unexpected names are a data-contract change.
KNOWN_REGIONS = {
    "Cherkaska oblast", "Chernihivska oblast", "Chernivetska oblast",
    "Dnipropetrovska oblast", "Donetska oblast", "Ivano-Frankivska oblast",
    "Kharkivska oblast", "Khersonska oblast", "Khmelnytska oblast",
    "Kirovohradska oblast", "Kyiv City", "Kyivska oblast",
    "Luhanska oblast", "Lvivska oblast", "Mykolaivska oblast",
    "Odeska oblast", "Poltavska oblast", "Rivnenska oblast",
    "Sumska oblast", "Ternopilska oblast", "Vinnytska oblast",
    "Volynska oblast", "Zakarpatska oblast", "Zaporizka oblast",
    "Zhytomyrska oblast",
}


class DataQualityError(ValueError):
    """The input file is unsuitable for calculation; tell the user why."""


def clean_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate rows; exclude invalid intervals and exact duplicates with an audit."""
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise DataQualityError(f"Відсутні стовпці CSV: {', '.join(missing)}")
    if raw.empty:
        raise DataQualityError("CSV порожній.")

    df = raw[["region", "started_at", "finished_at", "naive"]].copy()
    total = len(df)
    df["region"] = df["region"].astype("string").str.strip()
    bad_region = df["region"].isna() | df["region"].eq("")

    # Use UTC for comparisons; convert to Kyiv only to decide calendar days.
    for name in ("started_at", "finished_at"):
        df[name] = pd.to_datetime(df[name], utc=True, errors="coerce")
    bad_timestamp = df["started_at"].isna() | df["finished_at"].isna()

    flag = df["naive"].astype("string").str.strip().str.lower()
    df["naive"] = flag.map({"true": True, "false": False, "1": True, "0": False})
    bad_flag = df["naive"].isna()

    # Include equality: a zero-length interval carries no minutes of alert.
    bad_interval = (~bad_timestamp) & (df["finished_at"] <= df["started_at"])
    invalid = bad_region | bad_timestamp | bad_flag | bad_interval
    report = {
        "input_rows": int(total),
        "empty_region_rows": int(bad_region.sum()),
        "bad_timestamp_rows": int(bad_timestamp.sum()),
        "bad_naive_rows": int(bad_flag.sum()),
        "nonpositive_interval_rows": int(bad_interval.sum()),
        "invalid_rows_removed": int(invalid.sum()),
    }
    # Do not silently fabricate outputs from a severely malformed input file.
    if invalid.mean() > 0.01:
        raise DataQualityError(
            f"Понад 1% некоректних записів ({invalid.sum()} із {total}); "
            "перевір файл перед обробкою."
        )

    df = df.loc[~invalid].copy()
    unexpected = sorted(set(df["region"]) - KNOWN_REGIONS)
    if unexpected:
        raise DataQualityError(
            "Невідомі регіони: " + ", ".join(unexpected)
            + ". Не підміняємо їх іншими без перевірки."
        )

    duplicate = df.duplicated(subset=["region", "started_at", "finished_at", "naive"])
    report["exact_duplicate_rows_removed"] = int(duplicate.sum())
    df = df.loc[~duplicate].sort_values(["region", "started_at"]).reset_index(drop=True)
    df["naive"] = df["naive"].astype(bool)
    report["valid_rows"] = int(len(df))
    report["naive_true_rows"] = int(df["naive"].sum())
    if df.empty:
        raise DataQualityError("Не залишилося коректних інтервалів.")
    return df, report


def load_records(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read original CSV and run all validation rules."""
    path = Path(csv_path)
    if not path.is_file():
        raise DataQualityError(f"Не знайдено файл: {path}")
    try:
        raw = pd.read_csv(path, dtype="string")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataQualityError(f"Не вдалося прочитати CSV: {exc}") from exc
    return clean_records(raw)


def suggested_last_day(records: pd.DataFrame, today: date | None = None) -> date:
    """Be conservative: omit the last day with a start (may be incomplete)."""
    if today is None:
        today = datetime.now(KYIV).date()
    last_start = records["started_at"].max().to_pydatetime().astimezone(KYIV).date()
    return min(today - timedelta(days=1), last_start - timedelta(days=1))


def _union_hours(intervals: list[tuple[datetime, datetime]]) -> float:
    """Duration of union: overlapping alerts in one region are not double-counted."""
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    start, end = intervals[0]
    total_seconds = 0.0
    for current_start, current_end in intervals[1:]:
        if current_start <= end:
            if current_end > end:
                end = current_end
        else:
            total_seconds += (end - start).total_seconds()
            start, end = current_start, current_end
    total_seconds += (end - start).total_seconds()
    return total_seconds / 3600.0


def _split_by_kyiv_day(start: datetime, end: datetime):
    """Yield (Kyiv date, UTC start, UTC end); handles midnight and DST properly."""
    cursor = start.astimezone(UTC)
    end_utc = end.astimezone(UTC)
    while cursor < end_utc:
        local_day = cursor.astimezone(KYIV).date()
        # Crucial: *local* midnight converted to UTC, not UTC + 24 hours.
        next_midnight = datetime.combine(
            local_day + timedelta(days=1), time.min, tzinfo=KYIV
        ).astimezone(UTC)
        segment_end = min(next_midnight, end_utc)
        if segment_end <= cursor:
            raise DataQualityError("Помилка межі доби / часового поясу.")
        yield local_day, cursor, segment_end
        cursor = segment_end


def daily_hours(records: pd.DataFrame, through: date) -> pd.DataFrame:
    """Dense region-day table of recorded hours, NOT proof of zero-risk days.

    `without_naive_hours` and `naive_added_hours` quantify sensitivity to
    removing uncertain records. Overlap is correctly accounted for in both.
    """
    if records.empty:
        raise DataQualityError("Немає інтервалів для обробки.")
    first_day = records["started_at"].min().to_pydatetime().astimezone(KYIV).date()
    if through < first_day:
        raise DataQualityError("Кінцева дата передує першому запису.")
    stop_utc = datetime.combine(through + timedelta(days=1), time.min, tzinfo=KYIV).astimezone(UTC)

    grouped_all: dict[tuple[str, date], list] = defaultdict(list)
    grouped_confirmed: dict[tuple[str, date], list] = defaultdict(list)
    first_by_region: dict[str, date] = {}

    for region, start_ts, end_ts, is_naive in records[
        ["region", "started_at", "finished_at", "naive"]
    ].itertuples(index=False, name=None):
        start, end = start_ts.to_pydatetime(), end_ts.to_pydatetime()
        # A future start is invisible to this report.
        if start >= stop_utc:
            continue
        local_start_day = start.astimezone(KYIV).date()
        first_by_region[region] = min(first_by_region.get(region, local_start_day), local_start_day)
        for day, a, b in _split_by_kyiv_day(start, min(end, stop_utc)):
            key = (region, day)
            grouped_all[key].append((a, b))
            if not is_naive:
                grouped_confirmed[key].append((a, b))

    rows = []
    for region in sorted(first_by_region):
        # Never back-fill a region with artificial zeroes before its first record.
        start_day = first_by_region[region]
        for day in pd.date_range(start_day, through, freq="D"):
            day = day.date()
            key = (region, day)
            hours = _union_hours(grouped_all[key])
            confirmed = _union_hours(grouped_confirmed[key])
            if confirmed > hours + 1e-8:
                raise DataQualityError("Непослідовна агрегація інтервалів.")
            rows.append({
                "region": region,
                "date": day.isoformat(),
                "recorded_hours": round(hours, 6),
                "without_naive_hours": round(confirmed, 6),
                "naive_added_hours": round(max(0, hours - confirmed), 6),
                # 0 means no intervals in the file; this is NOT verified absence.
                "has_source_interval": bool(grouped_all[key]),
            })
    return pd.DataFrame(rows, columns=[
        "region", "date", "recorded_hours", "without_naive_hours",
        "naive_added_hours", "has_source_interval",
    ])
