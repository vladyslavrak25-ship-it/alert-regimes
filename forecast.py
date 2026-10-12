"""Stage 3: honest multi-horizon probability forecasts of recorded activity.

A forecast for day T made at origin O uses data <= O, never data from O+1...T.
Fixed region thresholds were defined on training history ending 2026-09-11.
This is research about *recorded alert durations*, not an operational safety service.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import math

import numpy as np
import pandas as pd
from pure_numpy_logistic import NumPyMultinomialLogistic

from regimes import (
    HOLDOUT_END, HOLDOUT_START, TRAIN_END,
    RegimeDataError, classify_hours, read_daily_table,
)

CLASSES = ("low", "medium", "high")
# Supports dates 12--18 October with an origin of 9 October (horizons 3--9).
MAX_HORIZON = 9
LOOKBACK = 28
# Tuning-free starting point (fixed before inspecting holdout scores).
TRAIN_DAYS = 365
# All model choices happen before the Sep12-Oct9 holdout.
SELECTION_END = date(2026, 8, 14)
SELECTION_TARGET_START = date(2026, 8, 15)
SELECTION_TARGET_END = date(2026, 9, 11)
SELECTION_ORIGINS = (date(2026, 8, 14), date(2026, 8, 21),
                     date(2026, 8, 28), date(2026, 9, 4))
TEST_ORIGINS = (date(2026, 9, 11), date(2026, 9, 18),
                date(2026, 9, 25), date(2026, 10, 2))


class ForecastError(ValueError):
    pass


def get_region_series(daily: pd.DataFrame, region: str, as_of: date) -> pd.Series:
    """Strictly require a full calendar from the first present date through cutoff."""
    sub = daily.loc[(daily["region"] == region) & (daily["date"] <= as_of),
                    ["date", "recorded_hours"]].sort_values("date")
    if sub.empty or sub.iloc[-1]["date"] != as_of:
        raise ForecastError(f"{region}: немає даних за дату завершення {as_of}.")
    if sub["date"].duplicated().any():
        raise ForecastError(f"{region}: повтори календарних дат.")
    expected = pd.date_range(sub.iloc[0]["date"], as_of, freq="D").date
    missing = set(expected) - set(sub["date"])
    if missing:
        raise ForecastError(f"{region}: відсутній календарний рядок за {min(missing)}.")
    series = pd.Series(sub["recorded_hours"].to_numpy(dtype=float),
                       index=pd.DatetimeIndex(sub["date"]))
    if not np.isfinite(series.to_numpy()).all() or (series < 0).any() or (series > 24.000001).any():
        raise ForecastError(f"{region}: некоректні добові години.")
    return series


def labels_from_hours(hours: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.array([classify_hours(float(h), low, high) for h in hours])


def baseline_probabilities(history: pd.Series, low: float, high: float) -> np.ndarray:
    """Recent 28-day frequencies with small Laplace smoothing to avoid zero estimates."""
    if len(history) < LOOKBACK:
        raise ForecastError("Замало попередніх днів для baseline.")
    labels = labels_from_hours(history.iloc[-LOOKBACK:].to_numpy(), low, high)
    counts = np.array([(labels == cls).sum() for cls in CLASSES], dtype=float)
    return (counts + 1) / (LOOKBACK + len(CLASSES))


def features_from_origin(history: pd.Series, target: date, horizon: int,
                         low: float, high: float) -> np.ndarray:
    """Only history up to origin and known calendar properties of target day."""
    if not 1 <= horizon <= MAX_HORIZON:
        raise ForecastError(f"Горизонт повинен бути від 1 до {MAX_HORIZON} днів.")
    if len(history) < LOOKBACK:
        raise ForecastError("Потрібні принаймні 28 минулих повних днів.")
    origin = history.index[-1].date()
    if origin + timedelta(days=horizon) != target:
        raise ForecastError("Дата прогнозу не відповідає даті зрізу та горизонту.")
    v = history.to_numpy(dtype=float)
    recent7 = v[-7:]
    recent28 = v[-28:]
    day_of_week = target.weekday()
    month = target.month
    high_count7 = float((recent7 > high).sum())
    return np.array([
        v[-1], float(recent7.mean()), float(recent28.mean()),
        float(recent7.std()), high_count7,
        float(recent7.mean() - recent28.mean()),
        math.sin(2*math.pi*day_of_week/7), math.cos(2*math.pi*day_of_week/7),
        math.sin(2*math.pi*(month-1)/12), math.cos(2*math.pi*(month-1)/12),
        float(horizon),
    ], dtype=float)


def fit_probabilistic_model(series: pd.Series, low: float, high: float,
                            as_of: date):
    """Build training samples from targets <= as_of; every row has its own past-only origin."""
    available = series.loc[:pd.Timestamp(as_of)]
    if available.empty or available.index[-1].date() != as_of:
        raise ForecastError("Навчальний зріз не закінчується потрібною датою.")
    if len(available) < LOOKBACK + TRAIN_DAYS:
        raise ForecastError("Замало історії для 365 навчальних днів і 28 днів ознак.")
    # The same past-only features as features_from_origin, calculated in batches.
    # This reduces per-region training time significantly without changing dates.
    values = available.to_numpy(dtype=float)
    n = len(values)
    last_targets = np.arange(n - TRAIN_DAYS, n)
    windows28 = np.lib.stride_tricks.sliding_window_view(values, LOOKBACK)
    weekday = available.index.dayofweek.to_numpy()
    month = available.index.month.to_numpy()
    samples, labels = [], []
    for horizon in range(1, MAX_HORIZON + 1):
        origins = last_targets - horizon
        mask = origins >= LOOKBACK - 1
        valid_origins = origins[mask]
        targets = last_targets[mask]
        window = windows28[valid_origins - (LOOKBACK - 1)]
        recent7 = window[:, -7:]
        avg7 = recent7.mean(axis=1)
        avg28 = window.mean(axis=1)
        days = weekday[targets]
        months = month[targets]
        x_batch = np.column_stack([
            values[valid_origins], avg7, avg28,
            recent7.std(axis=1), (recent7 > high).sum(axis=1).astype(float),
            avg7 - avg28,
            np.sin(2 * np.pi * days / 7), np.cos(2 * np.pi * days / 7),
            np.sin(2 * np.pi * (months - 1) / 12),
            np.cos(2 * np.pi * (months - 1) / 12),
            np.full(len(targets), float(horizon)),
        ])
        samples.append(x_batch)
        labels.extend(labels_from_hours(values[targets], low, high))
    y = np.array(labels)
    if set(y) != set(CLASSES):
        raise ForecastError("Навчальні приклади не містять усіх трьох категорій.")
    model = NumPyMultinomialLogistic(iterations=400, learning_rate=0.08, l2=0.003)
    model.fit(np.vstack(samples), y)
    return model


def predict_for_dates(model, series: pd.Series, as_of: date,
                      target_dates: list[date], low: float, high: float) -> pd.DataFrame:
    """Predict all dates at once from one origin; use no targets as features."""
    history = series.loc[:pd.Timestamp(as_of)]
    if history.empty or history.index[-1].date() != as_of:
        raise ForecastError("Не знайдено повну історію до потрібної дати зрізу.")
    rows = []
    base = baseline_probabilities(history, low, high)
    for target in target_dates:
        horizon = (target - as_of).days
        feature = features_from_origin(history, target, horizon, low, high)
        p = model.predict_proba(feature.reshape(1, -1))[0]
        mapped = dict(zip(model.classes_, p))
        ordered = np.array([mapped.get(c, 0.0) for c in CLASSES], dtype=float)
        if not np.isfinite(ordered).all() or np.any(ordered < -1e-10) or abs(ordered.sum()-1) > 1e-8:
            raise ForecastError("Модель повернула некоректні ймовірності.")
        rows.append({
            "date": target.isoformat(), "origin": as_of.isoformat(), "horizon": horizon,
            "low": ordered[0], "medium": ordered[1], "high": ordered[2],
            "baseline_low": base[0], "baseline_medium": base[1], "baseline_high": base[2],
        })
    return pd.DataFrame(rows)


def load_thresholds(path: str | Path) -> pd.DataFrame:
    f = Path(path)
    if not f.exists():
        raise ForecastError("Немає outputs/regime_thresholds.csv: запусти python run_regimes.py")
    df = pd.read_csv(f)
    needed = {"region", "q33_hours", "q75_hours", "fit_end"}
    if not needed.issubset(df.columns):
        raise ForecastError("У файлі порогів відсутні потрібні стовпці.")
    if len(df) != len(df["region"].unique()):
        raise ForecastError("Повторні області в порогах.")
    if not (df["fit_end"] == TRAIN_END.isoformat()).all():
        raise ForecastError("Пороги не зафіксовано на 11 вересня 2026 року.")
    if ((df["q33_hours"] <= 0) | (df["q75_hours"] <= df["q33_hours"])).any():
        raise ForecastError("Некоректні пороги.")
    return df


def multiclass_brier(p: np.ndarray, truth: str) -> float:
    """Sum of squared class-probability errors, range 0..2 (unscaled Brier)."""
    if truth not in CLASSES:
        raise ForecastError("Невідомий клас істини.")
    p = np.asarray(p, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum()-1) > 1e-8:
        raise ForecastError("Некоректний вектор ймовірностей.")
    actual = np.array([float(c == truth) for c in CLASSES])
    return float(((p - actual)**2).sum())



# Stage 3 correction: one probability forecast for every (region, target day)
# in each experiment. Models are refit at *each* weekly forecast origin.


def selection_thresholds(series: pd.Series) -> tuple[float, float]:
    """Thresholds for PRE-holdout model selection: do not use Sep 12+ or Sep 11.

    The 180-day selection reference ends Aug 14. These thresholds are for
    deciding which algorithm generalizes better in Aug15-Sep11 ONLY; the
    production/holdout thresholds remain frozen from Stage 2 through Sep 11.
    """
    reference = series.loc[:pd.Timestamp(SELECTION_END)].iloc[-180:]
    if len(reference) != 180 or reference.index[-1].date() != SELECTION_END:
        raise ForecastError("Недостатньо повної історії до 14 серпня для вибору моделі.")
    low = float(reference.quantile(0.33))
    high = float(reference.quantile(0.75))
    if low <= 0 or high <= low:
        raise ForecastError("Неможливо виділити три категорії для попередньої валідації.")
    return low, high


def evaluate_experiment(
    series: pd.Series,
    low: float,
    high: float,
    origins: tuple[date, ...],
    start: date,
    end: date,
    first_horizon: int,
    experiment: str,
    model_cache: dict[date, object] | None = None,
) -> pd.DataFrame:
    """Nonoverlapping weekly issue dates, same history cutoff for both models.

    first_horizon=1 evaluates a conventional 7-day forecast.
    first_horizon=3 evaluates forecasts issued with a two-day gap.
    Only complete historical targets within [start, end] are scored.
    """
    if first_horizon not in (1, 3):
        raise ForecastError("Дозволено починати прогноз лише з горизонту 1 або 3.")
    if not origins or sorted(origins) != list(origins):
        raise ForecastError("Дати створення прогнозів мають іти за часом.")
    if model_cache is None:
        model_cache = {}
    seen = set()
    rows = []
    for origin in origins:
        if origin >= start and origin >= end:
            raise ForecastError("Некоректна дата прогнозу.")
        forecast_dates = [origin + timedelta(days=h)
                          for h in range(first_horizon, first_horizon + 7)
                          if start <= origin + timedelta(days=h) <= end]
        if not forecast_dates:
            continue
        if origin not in model_cache:
            model_cache[origin] = fit_probabilistic_model(series, low, high, origin)
        forecasts = predict_for_dates(
            model_cache[origin], series, origin, forecast_dates, low, high
        )
        for row in forecasts.itertuples(index=False):
            if row.date in seen:
                raise ForecastError(f"Повторна прогнозована дата {row.date} у {experiment}.")
            seen.add(row.date)
            actual = classify_hours(float(series.loc[pd.Timestamp(row.date)]), low, high)
            model_p = np.array([row.low, row.medium, row.high])
            baseline_p = np.array([row.baseline_low, row.baseline_medium, row.baseline_high])
            rows.append({
                "experiment": experiment,
                "date": row.date, "origin": row.origin, "horizon": row.horizon,
                "actual_regime": actual,
                "model_low": row.low, "model_medium": row.medium, "model_high": row.high,
                "baseline_low": row.baseline_low,
                "baseline_medium": row.baseline_medium,
                "baseline_high": row.baseline_high,
                "model_brier": multiclass_brier(model_p, actual),
                "baseline_brier": multiclass_brier(baseline_p, actual),
            })
    if not rows:
        raise ForecastError(f"У {experiment} немає прогнозів для оцінювання.")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def choose_model(validation_predictions: pd.DataFrame) -> str:
    """ONE choice made on pre-holdout validation only; ties go to baseline."""
    if validation_predictions.empty:
        raise ForecastError("Немає даних валідації для вибору моделі.")
    for col in ("model_brier", "baseline_brier"):
        if not np.isfinite(validation_predictions[col].to_numpy(dtype=float)).all():
            raise ForecastError("Некоректні результати валідації.")
    return ("logistic" if validation_predictions["model_brier"].mean()
            < validation_predictions["baseline_brier"].mean() else "baseline")


def evaluate_region(
    series: pd.Series, low: float, high: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Return validation B, independent holdout A/B and PRE-selected model.

    Validation data: Aug15-Sep11. Thresholds for it: 180 days ending Aug14.
    Holdout A: Sep12-Oct9, horizons 1..7, 28 unique target days.
    Holdout B: Sep14-Oct9, horizons 3..9, 26 unique target days.
    Two holdout experiments share some dates and are NOT independent samples.
    """
    vlow, vhigh = selection_thresholds(series)
    val = evaluate_experiment(
        series, vlow, vhigh, SELECTION_ORIGINS, SELECTION_TARGET_START,
        SELECTION_TARGET_END, 3, "validation_gap_3_9"
    )
    choice = choose_model(val)
    cache = {}
    a = evaluate_experiment(series, low, high, TEST_ORIGINS,
                            HOLDOUT_START, HOLDOUT_END, 1,
                            "holdout_next_1_7", model_cache=cache)
    b = evaluate_experiment(series, low, high, TEST_ORIGINS,
                            HOLDOUT_START, HOLDOUT_END, 3,
                            "holdout_gap_3_9", model_cache=cache)
    if not (len(val) == 26 and len(a) == 28 and len(b) == 26):
        raise ForecastError(f"Неочікувана кількість днів: val={len(val)}, A={len(a)}, B={len(b)}")
    if a["date"].nunique() != 28 or b["date"].nunique() != 26:
        raise ForecastError("У фінальному тесті повторюються календарні дати.")
    return val, a, b, choice


def selected_probabilities(predictions: pd.DataFrame, choice: str) -> pd.DataFrame:
    """Provide one chosen set of probabilities, keeping both for audit."""
    if choice not in ("baseline", "logistic"):
        raise ForecastError("Невідома обрана модель.")
    result = predictions.copy()
    prefix = "baseline" if choice == "baseline" else "model"
    for cls in CLASSES:
        result[f"selected_{cls}"] = result[f"{prefix}_{cls}"]
    result["selected_model"] = choice
    if "actual_regime" in result:
        result["selected_brier"] = result[f"{prefix}_brier"]
    return result


def read_and_validate_inputs(daily_file: str | Path, threshold_file: str | Path):
    daily = read_daily_table(daily_file)
    thresholds = load_thresholds(threshold_file)
    if not daily["date"].ge(HOLDOUT_END).any():
        raise ForecastError("Потрібен добовий файл із датами до 9 жовтня включно.")
    return daily, thresholds
