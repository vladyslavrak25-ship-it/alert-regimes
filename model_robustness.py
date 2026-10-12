"""EXPLORATORY earlier rolling-origin comparison, not final forecast selection.

Never runs or rewrites run_forecast.py; writes to `analysis/` only.
For a fold, thresholds are frozen at its FIRST origin, before every target.
Each weekly origin then fits logistic only on its own available history.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from forecast import (
    ForecastError, evaluate_experiment, get_region_series,
)

TARGET_REGIONS = (
    'Chernihivska oblast', 'Sumska oblast', 'Zaporizka oblast',
)
# Four non-overlapping 28-day TARGET periods; the last target is Aug 9.
# All precede Aug 15 (the existing selection window) and Sep 12 holdout.
FIRST_ORIGINS = (
    date(2026, 4, 17), date(2026, 5, 15),
    date(2026, 6, 12), date(2026, 7, 10),
)
SELECTION_START = date(2026, 8, 15)


def fold_geometry(first_origin: date) -> tuple[tuple[date, ...], date, date]:
    origins = tuple(first_origin + timedelta(days=7 * k) for k in range(4))
    target_start = first_origin + timedelta(days=3)
    target_end = origins[-1] + timedelta(days=9)
    if target_end >= SELECTION_START:
        raise ForecastError('Додатковий експеримент не повинен торкатися валідації 15.08+.')
    return origins, target_start, target_end


def historical_thresholds(series: pd.Series, first_origin: date) -> tuple[float, float]:
    """Exactly 180 days BEFORE scored days. Future edits cannot change them."""
    prev = series.loc[:pd.Timestamp(first_origin)]
    if prev.empty or prev.index[-1].date() != first_origin:
        raise ForecastError('Немає історії до точки заморожування порогів.')
    previous_180 = prev.iloc[-180:]
    if len(previous_180) != 180:
        raise ForecastError('Замало попередніх днів для порогів.')
    low, high = float(previous_180.quantile(.33)), float(previous_180.quantile(.75))
    if not 0 < low < high:
        raise ForecastError('Пороги не розділяють три класи.')
    return low, high


def compute_fold(series: pd.Series, region: str, first_origin: date) -> tuple[pd.DataFrame, dict]:
    origins, start, end = fold_geometry(first_origin)
    low, high = historical_thresholds(series, first_origin)
    frame = evaluate_experiment(
        series, low, high, origins, start, end, first_horizon=3,
        experiment=f'exploratory_{first_origin:%Y-%m-%d}',
    )
    if len(frame) != 28 or frame['date'].nunique() != 28:
        raise ForecastError('Очікувалося рівно 28 неповторних прогнозованих дат.')
    if (pd.to_datetime(frame['date']).dt.date <= first_origin).any():
        raise ForecastError('Знайдено прогноз на минуле.')
    frame.insert(0, 'region', region)
    frame.insert(1, 'reference_end', first_origin.isoformat())
    frame.insert(2, 'reference_days', 180)
    frame.insert(3, 'reference_q33', low)
    frame.insert(4, 'reference_q75', high)
    base = float(frame['baseline_brier'].mean())
    logistic = float(frame['model_brier'].mean())
    summary = dict(region=region, reference_end=first_origin.isoformat(),
                   first_target=start.isoformat(), last_target=end.isoformat(),
                   target_days=len(frame), q33_hours=low, q75_hours=high,
                   baseline_brier=base, logistic_brier=logistic,
                   logistic_minus_baseline=logistic-base,
                   lower_brier='logistic' if logistic < base else 'baseline')
    return frame, summary


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty or folds[['baseline_brier','logistic_brier']].isna().any().any():
        raise ForecastError('Порожні або пошкоджені оцінки.')
    records = []
    for region, chunk in folds.groupby('region', sort=False):
        n = int(chunk['target_days'].sum())
        baseline = float(np.average(chunk['baseline_brier'], weights=chunk['target_days']))
        model = float(np.average(chunk['logistic_brier'], weights=chunk['target_days']))
        records.append(dict(region=region, folds=len(chunk), target_days=n,
                            baseline_brier=baseline, logistic_brier=model,
                            logistic_minus_baseline=model-baseline,
                            folds_logistic_better=int((chunk['logistic_brier']<chunk['baseline_brier']).sum()),
                            exploratory_winner='logistic' if model < baseline else 'baseline'))
    return pd.DataFrame(records)


def run_exploration(daily: pd.DataFrame, region_names=TARGET_REGIONS,
                    origins=FIRST_ORIGINS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not origins or len(set(origins)) != len(origins) or tuple(origins) != tuple(sorted(origins)):
        raise ForecastError('Некоректні або дубльовані точки експерименту.')
    periods = [fold_geometry(d) for d in origins]
    for i in range(len(periods)-1):
        if periods[i][2] >= periods[i+1][1]:
            raise ForecastError('Перекриття цільових дат між експериментами.')
    prediction_frames, fold_rows = [], []
    for region in region_names:
        as_of = max(period[2] for period in periods)
        series = get_region_series(daily, region, as_of)
        for first_origin in origins:
            frame, row = compute_fold(series, region, first_origin)
            prediction_frames.append(frame)
            fold_rows.append(row)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    summary = summarize_folds(folds)
    if predictions.duplicated(['region','date']).any():
        raise ForecastError('В експерименті повторюється цільова дата для області.')
    return predictions, folds, summary
