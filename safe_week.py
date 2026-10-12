"""Safely forecast seven consecutive Kyiv calendar days from a frozen past.

This is a retrospective reconstruction unless publication times prove that the
source snapshot was available BEFORE target_start (not available in Volunteer CSV).
Historical training/selection thresholds remain frozen on pre-holdout data.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data import KYIV, daily_hours
from forecast import (
    CLASSES, ForecastError, baseline_probabilities, fit_probabilistic_model,
    get_region_series, load_thresholds, predict_for_dates,
)
from regimes import TRAIN_END

MIN_TARGET_START = TRAIN_END + timedelta(days=1)
CHOICE_COLS = [
    "region", "selected_model", "selection_end", "selection_gap_brier_logistic",
    "selection_gap_brier_baseline", "selection_forecasts",
]


def midnight_utc(target_start: date) -> datetime:
    """First forbidden instant, expressed in UTC; Kyiv DST handled correctly."""
    return datetime.combine(target_start, time.min, tzinfo=KYIV).astimezone(timezone.utc)


def frozen_past(records: pd.DataFrame, target_start: date) -> pd.DataFrame:
    """Exclude starts on/after target_start and clip ends crossing its midnight.

    The output cannot depend on the contents of any event which *starts* on/after
    the boundary, nor on the post-boundary part of a crossing event.
    """
    if not isinstance(target_start, date) or target_start < MIN_TARGET_START:
        raise ForecastError("Потрібен початок після фіксації порогів 11.09.2026.")
    needed = {"region", "started_at", "finished_at", "naive"}
    if not needed.issubset(records.columns):
        raise ForecastError("У записах відсутні поля Volunteer CSV.")
    boundary = pd.Timestamp(midnight_utc(target_start))
    old = records.loc[records["started_at"] < boundary].copy()
    if old.empty:
        raise ForecastError("Немає історії до прогнозного періоду.")
    old["finished_at"] = old["finished_at"].clip(upper=boundary)
    if not (old["started_at"] < old["finished_at"]).all():
        raise ForecastError("Після часової межі залишився недопустимий інтервал.")
    assert (old["started_at"] < boundary).all()
    assert (old["finished_at"] <= boundary).all()
    return old


def frozen_fingerprint(records: pd.DataFrame, target_start: date) -> str:
    """Canonical digest of ONLY accepted pre-forecast information."""
    f = frozen_past(records, target_start)
    cols = ["region", "started_at", "finished_at", "naive"]
    # Stable sorting independent of input order; UTC timestamps in text form.
    text = f[cols].astype(str).sort_values(cols, kind="mergesort").to_csv(index=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_choices(path: str | Path) -> pd.DataFrame:
    """Read normal Stage 3 CSV and the observed headerless six-column export."""
    p = Path(path)
    if not p.is_file():
        raise ForecastError(f"Відсутній файл вибору моделей: {p}")
    frame = pd.read_csv(p, encoding="utf-8-sig")
    if not set(CHOICE_COLS).issubset(frame.columns):
        frame = pd.read_csv(p, header=None, names=CHOICE_COLS, encoding="utf-8-sig")
    if frame.empty or frame["region"].duplicated().any():
        raise ForecastError("Некоректний файл вибору моделей (порожній/повторні регіони).")
    if not frame["selected_model"].isin(["baseline", "logistic"]).all():
        raise ForecastError("Невідома обрана модель.")
    if (frame["selection_end"].astype(str) != TRAIN_END.isoformat()).any():
        raise ForecastError("Вибір моделей повинен бути завершений до прогнозу.")
    return frame


def seven_dates(target_start: date) -> list[date]:
    if not isinstance(target_start, date) or target_start < MIN_TARGET_START:
        raise ForecastError("Дата початку передує завершенню вибору моделей.")
    return [target_start + timedelta(days=i) for i in range(7)]


def forecast_from_records(
    records: pd.DataFrame,
    thresholds: pd.DataFrame,
    choices: pd.DataFrame,
    target_start: date,
    as_of: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate chosen probabilities, without ever passing future rows to ML.

    No model selection is repeated. Thresholds/choices were frozen in Sept 2026.
    Returns (forecast, accepted_region_daily). Neither is saved automatically.
    """
    dates = seven_dates(target_start)
    if as_of is None:
        as_of = target_start - timedelta(days=1)
    if as_of < TRAIN_END or as_of >= target_start:
        raise ForecastError("Історичний зріз має бути після 11.09.2026 та ДО початку прогнозу.")
    # The *last known* day may be earlier than the forecast week.
    # Never let a Sunday partial record leak into a Monday forecast.
    past = frozen_past(records, as_of + timedelta(days=1))
    daily = daily_hours(past, through=as_of)
    # Stage 1 emits ISO string dates; Stage 3 expects Python date objects.
    daily_for_model = daily.copy()
    daily_for_model["date"] = pd.to_datetime(daily["date"], format="%Y-%m-%d").dt.date
    if len(thresholds) != len(choices) or set(thresholds["region"]) != set(choices["region"]):
        raise ForecastError("Регіони у порогах і виборі моделі не збігаються.")
    if not (thresholds["fit_end"].astype(str) == TRAIN_END.isoformat()).all():
        raise ForecastError("Пороги повинні бути заморожені на 11.09.2026.")
    if (choices["selection_end"].astype(str) != TRAIN_END.isoformat()).any():
        raise ForecastError("Вибір моделей не був зафіксований до тестового періоду.")
    selection = choices.set_index("region")["selected_model"].to_dict()
    future = []
    for th in thresholds.itertuples(index=False):
        region = th.region
        lo, hi = float(th.q33_hours), float(th.q75_hours)
        if not 0 < lo < hi:
            raise ForecastError("Некоректні межі категорій.")
        series = get_region_series(daily_for_model, region, as_of)
        choice = selection[region]
        if choice == "baseline":
            p = baseline_probabilities(series, lo, hi)
            raw_predictions = [p] * 7
        elif choice == "logistic":
            trained = fit_probabilistic_model(series, lo, hi, as_of)
            pred = predict_for_dates(trained, series, as_of, dates, lo, hi)
            raw_predictions = pred[list(CLASSES)].to_numpy()
        else:
            raise ForecastError("Некоректна обрана модель.")
        for d, probs in zip(dates, raw_predictions):
            p = np.asarray(probs, dtype=float)
            if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 1e-7:
                raise ForecastError("Некоректні ймовірності прогнозу.")
            future.append({
                "region": region,
                "date": d.isoformat(),
                "origin": as_of.isoformat(),
                "horizon": (d - as_of).days,
                "selected_model": choice,
                "selected_low": float(p[0]),
                "selected_medium": float(p[1]),
                "selected_high": float(p[2]),
            })
    result = pd.DataFrame(future)
    if result.duplicated(["region", "date"]).any():
        raise ForecastError("Повторна область/дата в прогнозі.")
    expected_horizons = {(d - as_of).days for d in dates}
    if set(result["horizon"]) != expected_horizons:
        raise ForecastError("Некоректні горизонти прогнозу.")
    if not set(daily["date"]).isdisjoint(dates):
        raise ForecastError("Витік: прогнозні дати містяться у навчальній таблиці.")
    return result, daily.loc[daily["region"].isin(selection)].copy()


def save_new_run(
    csv_path: Path,
    target_start: date,
    forecast: pd.DataFrame,
    past_daily: pd.DataFrame,
    records: pd.DataFrame,
    outdir: Path,
    thresholds_file: Path,
    choices_file: Path,
) -> list[Path]:
    """Never silently replace a previous run or the Stage 3 outputs."""
    if outdir.exists():
        raise ForecastError(f"Папка результату вже існує: {outdir}. Обери іншу назву.")
    outdir.mkdir(parents=True, exist_ok=False)
    last_start = records["started_at"].max().to_pydatetime().astimezone(KYIV)
    midnight = pd.Timestamp(midnight_utc(target_start))
    crossing = int(((records["started_at"] < midnight) & (records["finished_at"] > midnight)).sum())
    excluded = int((records["started_at"] >= midnight).sum())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = {
        "forecast_first_day": target_start.isoformat(),
        "forecast_last_day": (target_start + timedelta(days=6)).isoformat(),
        "last_training_day": (target_start - timedelta(days=1)).isoformat(),
        "forbidden_from_kyiv": f"{target_start.isoformat()}T00:00:00 Europe/Kyiv",
        "horizons": list(range(1, 8)),
        "created_at_kyiv": datetime.now(KYIV).isoformat(),
        "source_path": str(csv_path),
        "source_sha256": sha(csv_path),
        "source_latest_start_kyiv": last_start.isoformat(),
        "records_started_on_or_after_boundary_excluded": excluded,
        "intervals_crossing_boundary_clipped": crossing,
        "accepted_past_sha256": frozen_fingerprint(records, target_start),
        "thresholds_sha256": sha(thresholds_file),
        "model_selection_sha256": sha(choices_file),
        "fixed_threshold_fit_end": TRAIN_END.isoformat(),
        "model_choice_policy": "Existing Sep11 validation choices, not reselected on future data",
        "interpretation": "Recorded alert-duration categories, NOT attack/safety predictions",
        "source_availability": "Retrospective snapshot; this CSV has no per-record publication times. Do not claim it was available at forecast origin.",
        "note": "New run is separate from frozen original Oct9 forecast and does not replace Streamlit outputs.",
    }
    path_forecast = outdir / "forecast_7days.csv"
    path_daily = outdir / "accepted_past_daily.csv"
    path_manifest = outdir / "run_manifest.json"
    forecast.to_csv(path_forecast, index=False, encoding="utf-8-sig")
    past_daily.to_csv(path_daily, index=False, encoding="utf-8-sig")
    path_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return [path_forecast, path_daily, path_manifest]
