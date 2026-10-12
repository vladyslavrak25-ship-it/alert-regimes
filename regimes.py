"""Stage 2: choose activity regimes using ONLY a historical training period.

This module creates research labels for *recorded alert duration*, not safety
levels. Future/test days must not influence category thresholds or region choice.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

TRAIN_START = date(2026, 3, 16)
TRAIN_END = date(2026, 9, 11)
HOLDOUT_START = date(2026, 9, 12)
HOLDOUT_END = date(2026, 10, 9)

# Agreed shortlist. Changes here must be decided before examining test accuracy.
CANDIDATE_REGIONS = (
    "Chernihivska oblast",
    "Kharkivska oblast",
    "Sumska oblast",
    "Zaporizka oblast",
    "Dnipropetrovska oblast",
    "Poltavska oblast",
    "Mykolaivska oblast",
)

# Explicit preliminary suitability rules; based on TRAINING data only.
MIN_DAYS_WITH_INTERVAL = 175  # out of 180 training days
MIN_MEAN_HOURS = 3.0
MAX_NAIVE_HOURS_SHARE = 0.10
MIN_THRESHOLD_GAP_HOURS = 0.5
MIN_DAYS_IN_EACH_CLASS = 20

REGIME_NAMES = {"low": "Низька", "medium": "Помірна", "high": "Висока"}
REQUIRED = {
    "region", "date", "recorded_hours", "naive_added_hours", "has_source_interval"
}


class RegimeDataError(ValueError):
    """A clear error instead of silently creating incorrect classes."""


def read_daily_table(path: str | Path) -> pd.DataFrame:
    """Read Stage 1's daily output; verify expected shape and types."""
    path = Path(path)
    if not path.is_file():
        raise RegimeDataError(
            f"Немає файлу {path}. Спочатку запусти: python run_data.py"
        )
    try:
        df = pd.read_csv(path, dtype={"region": "string", "date": "string"})
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise RegimeDataError(f"Помилка читання добової таблиці: {exc}") from exc
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise RegimeDataError(f"У таблиці відсутні стовпці: {', '.join(missing)}")
    if df.empty:
        raise RegimeDataError("Добова таблиця порожня.")
    if df[["region", "date"]].isna().any().any():
        raise RegimeDataError("Є порожня область або дата.")

    try:
        dates = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="raise")
    except (ValueError, TypeError) as exc:
        raise RegimeDataError(f"Неправильний формат дати: {exc}") from exc
    df["date"] = dates.dt.date

    for col in ("recorded_hours", "naive_added_hours"):
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise RegimeDataError(f"Нечислові значення в {col}: {exc}") from exc
        if df[col].isna().any() or (~df[col].between(0, 25.000001)).any():
            raise RegimeDataError(f"Недопустимі години у стовпці {col}.")

    if (df["naive_added_hours"] > df["recorded_hours"] + 0.00001).any():
        raise RegimeDataError("naive_added_hours перевищує recorded_hours.")
    interval_flag = df["has_source_interval"].astype("string").str.lower()
    if not interval_flag.isin(["true", "false"]).all():
        raise RegimeDataError("has_source_interval повинен бути true або false.")
    df["has_source_interval"] = interval_flag.eq("true").astype(bool)
    if ((~df["has_source_interval"]) & (df["recorded_hours"] > 0)).any():
        raise RegimeDataError("Знайдено години тривоги без позначки інтервалу.")
    if df.duplicated(["region", "date"]).any():
        raise RegimeDataError("Дублікат пари область + день у добовій таблиці.")
    return df


def training_only(
    daily: pd.DataFrame,
    regions: tuple[str, ...] = CANDIDATE_REGIONS,
    start: date = TRAIN_START,
    end: date = TRAIN_END,
) -> pd.DataFrame:
    """Keep only allowed historic days; refuse missing calendar rows."""
    if start > end:
        raise RegimeDataError("Початок навчання пізніший за кінець.")
    if not regions or len(regions) != len(set(regions)):
        raise RegimeDataError("Список областей порожній або з повтореннями.")

    train = daily.loc[
        daily["region"].isin(regions)
        & daily["date"].between(start, end)
    ].copy()

    expected_days = set(pd.date_range(start, end, freq="D").date)
    for region in regions:
        region_days = set(train.loc[train["region"] == region, "date"])
        missing = expected_days - region_days
        if missing:
            first_missing = min(missing)
            raise RegimeDataError(
                f"Для {region} відсутні {len(missing)} календарних рядків у "
                f"навчальному періоді (перший: {first_missing}). "
                "Це відсутні рядки таблиці, не підтверджені нульові дні."
            )
    return train.sort_values(["region", "date"]).reset_index(drop=True)


def classify_hours(hours: float, low_limit: float, high_limit: float) -> str:
    """Use three mutually exclusive categories (at exact borders: lower class)."""
    if low_limit < 0 or high_limit <= low_limit or not 0 <= hours <= 25.000001:
        raise RegimeDataError("Некоректна тривалість або пороги категорій.")
    if hours <= low_limit:
        return "low"
    if hours <= high_limit:
        return "medium"
    return "high"


def build_training_regimes(
    daily: pd.DataFrame,
    regions: tuple[str, ...] = CANDIDATE_REGIONS,
    start: date = TRAIN_START,
    end: date = TRAIN_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frozen thresholds, TRAIN-ONLY daily labels and suitability report.

    The last 28 days are never examined here. In particular, no threshold is
    calibrated on dates later than 'end'. The forecast stage will add them.
    """
    train = training_only(daily, regions, start, end)
    threshold_rows, label_parts, audit_rows = [], [], []

    for region in regions:
        part = train.loc[train["region"] == region].copy()
        low_limit = float(part["recorded_hours"].quantile(0.33))
        high_limit = float(part["recorded_hours"].quantile(0.75))
        mean_hours = float(part["recorded_hours"].mean())
        interval_days = int(part["has_source_interval"].sum())
        total_hours = float(part["recorded_hours"].sum())
        naive_share = float(part["naive_added_hours"].sum() / total_hours) if total_hours else 1.0

        # Do not attempt classification with identical thresholds.
        if high_limit > low_limit >= 0:
            part["regime"] = part["recorded_hours"].map(
                lambda h: classify_hours(float(h), low_limit, high_limit)
            )
            counts = part["regime"].value_counts().to_dict()
        else:
            counts = {}

        reasons = []
        if interval_days < MIN_DAYS_WITH_INTERVAL:
            reasons.append(f"лише {interval_days} днів із записаним інтервалом")
        if mean_hours < MIN_MEAN_HOURS:
            reasons.append(f"середня тривалість {mean_hours:.2f} < {MIN_MEAN_HOURS} год")
        if naive_share > MAX_NAIVE_HOURS_SHARE:
            reasons.append(f"частка умовних годин {naive_share:.1%} > {MAX_NAIVE_HOURS_SHARE:.0%}")
        if low_limit <= 0 or (high_limit - low_limit) < MIN_THRESHOLD_GAP_HOURS:
            reasons.append("пороги не розрізняють достатньо три рівні")
        if any(counts.get(label, 0) < MIN_DAYS_IN_EACH_CLASS for label in REGIME_NAMES):
            reasons.append("замало днів принаймні в одному з трьох класів")

        suitable = not reasons
        audit_rows.append({
            "region": region,
            "train_start": start.isoformat(),
            "train_end": end.isoformat(),
            "train_days": int(len(part)),
            "days_with_source_interval": interval_days,
            "mean_hours": round(mean_hours, 4),
            "naive_hours_share": round(naive_share, 4),
            "low_days": int(counts.get("low", 0)),
            "medium_days": int(counts.get("medium", 0)),
            "high_days": int(counts.get("high", 0)),
            "suitable": bool(suitable),
            "reason": "; ".join(reasons) if reasons else "Критерії виконані",
        })
        if suitable:
            threshold_rows.append({
                "region": region,
                "q33_hours": low_limit,  # keep full precision for future predictions
                "q75_hours": high_limit,
                "fit_start": start.isoformat(),
                "fit_end": end.isoformat(),
            })
            part["regime_uk"] = part["regime"].map(REGIME_NAMES)
            label_parts.append(part[[
                "region", "date", "recorded_hours", "regime", "regime_uk",
                "has_source_interval", "naive_added_hours",
            ]])

    if not label_parts:
        raise RegimeDataError("Жодна область не пройшла критерії придатності.")
    thresholds = pd.DataFrame(threshold_rows)
    labels = pd.concat(label_parts, ignore_index=True)
    labels["date"] = labels["date"].astype(str)
    report = pd.DataFrame(audit_rows)
    return thresholds, labels, report
