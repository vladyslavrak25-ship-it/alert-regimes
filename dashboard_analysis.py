"""Pure data-preparation helpers for the category-first / day-click dashboard.

These helpers never fit models and never alter precomputed forecasts.
"""
from __future__ import annotations

from datetime import date
import numpy as np
import pandas as pd

from dashboard_data import REGION_UK, REGIME_UK, DashboardDataError, classify_daily

CATS = ("low", "medium", "high")
PERIOD_START = date(2022, 2, 25)
PERIOD_END = date(2026, 10, 9)


def subset_all_regions(daily: pd.DataFrame, regions: list[str], start: date,
                       end: date, cutoff: date) -> pd.DataFrame:
    """Preserve all AVAILABLE region-days in selected historic range; no invented prehistory."""
    if start > end or end > cutoff:
        raise DashboardDataError("Некоректний історичний період або дати пізніше зрізу.")
    if len(regions) != len(set(regions)) or not regions:
        raise DashboardDataError("Потрібен непорожній унікальний список областей.")
    days = pd.to_datetime(daily["date"])
    mask = daily["region"].isin(regions) & days.dt.date.between(start, end)
    result = daily.loc[mask].copy()
    if result.empty:
        raise DashboardDataError("Для цього періоду немає історичних рядків.")
    if result.duplicated(["region", "date"]).any():
        raise DashboardDataError("Повторні пари область-дата в історії.")
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values(["region", "date"]).reset_index(drop=True)


def regional_category_counts(period: pd.DataFrame, thresholds: pd.DataFrame,
                             regions: list[str]) -> pd.DataFrame:
    """Classify using frozen thresholds (2026); counts must match observed calendar rows."""
    if thresholds["region"].duplicated().any():
        raise DashboardDataError("У порогах є повторна область.")
    lookup = thresholds.set_index("region")
    records = []
    for region in regions:
        if region not in lookup.index:
            raise DashboardDataError(f"Відсутні пороги для {region}.")
        one = period.loc[period["region"] == region]
        if one.empty:
            continue
        row = lookup.loc[region]
        q33, q75 = float(row["q33_hours"]), float(row["q75_hours"])
        labels = classify_daily(one["recorded_hours"], q33, q75)
        counts = labels.value_counts().reindex(CATS, fill_value=0)
        n = len(one)
        if int(counts.sum()) != n:
            raise DashboardDataError(f"Сума категорій не дорівнює кількості днів для {region}.")
        records.append({
            "region": region, "Область": REGION_UK.get(region, region),
            "Низька": int(counts["low"]), "Помірна": int(counts["medium"]),
            "Висока": int(counts["high"]), "Усього днів": n,
            "Поріг q33, год": q33, "Поріг q75, год": q75,
        })
    return pd.DataFrame(records)


def selected_day_forecast(future: pd.DataFrame, day: date, regions: list[str]) -> pd.DataFrame:
    """Every chosen date must provide exactly one forecast for every selected region."""
    one = future.loc[pd.to_datetime(future["date"]).dt.date == day].copy()
    if one["region"].duplicated().any() or set(one["region"]) != set(regions):
        raise DashboardDataError(f"Прогноз на {day} має містити рівно 7 неповторних областей.")
    one["Область"] = one["region"].map(REGION_UK)
    for cat in CATS:
        one[REGIME_UK[cat]] = pd.to_numeric(one[f"selected_{cat}"], errors="coerce") * 100.0
    probs = one[[REGIME_UK[c] for c in CATS]].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 100, atol=0.0001):
        raise DashboardDataError("Для одного з регіонів ймовірності не складають 100%.")
    return one


def verify_model_choices(future: pd.DataFrame, selection: pd.DataFrame) -> dict[str, int]:
    """Selected probabilities must be exactly the named candidate's probabilities."""
    cmap = selection.set_index("region")["selected_model"].to_dict()
    counts = {"baseline": 0, "logistic": 0}
    for region, model in cmap.items():
        if model not in counts:
            raise DashboardDataError(f"Невідома модель {model}.")
        counts[model] += 1
        rows = future.loc[future["region"] == region]
        if len(rows) != 7 or not (rows["selected_model"] == model).all():
            raise DashboardDataError(f"Несумісна обрана модель для {region}.")
        prefix = "baseline" if model == "baseline" else "model"
        for cat in CATS:
            actual = rows[f"selected_{cat}"].to_numpy(dtype=float)
            expected = rows[f"{prefix}_{cat}"].to_numpy(dtype=float)
            if not np.allclose(actual, expected, atol=1e-10):
                raise DashboardDataError(f"Некоректно обрані ймовірності {region}/{cat}.")
    return counts


def verify_frozen_windows(selection: pd.DataFrame, thresholds: pd.DataFrame,
                          manifest: dict) -> None:
    """Ensure the calibration/reference dates precede the targets scored with them."""
    if not (pd.to_datetime(selection["selection_end"]).dt.date == date(2026, 9, 11)).all():
        raise DashboardDataError("Несподіваний кінець попередньої валідації.")
    if not (pd.to_datetime(thresholds["fit_end"]).dt.date == date(2026, 9, 11)).all():
        raise DashboardDataError("Фінальні пороги більше не заморожені до 11.09.")
    if manifest.get("selection_threshold_reference_180days_through") != "2026-08-14":
        raise DashboardDataError("Попередні пороги повинні закінчуватися 14.08.")
    if manifest.get("selection_target_range") != ["2026-08-15", "2026-09-11"]:
        raise DashboardDataError("Неочікуване валідаційне вікно.")
    if manifest.get("final_thresholds_through") != "2026-09-11":
        raise DashboardDataError("Неочікувані пороги для фінального тесту.")
    if manifest.get("holdout_target_range") != ["2026-09-12", "2026-10-09"]:
        raise DashboardDataError("Неочікуваний період фінального тестування.")


def current_snapshot_warning(audit: dict, manifest: dict) -> str:
    """Transparency text, not a claim about remote data freshness."""
    origin = manifest["date_of_last_complete_observation"]
    latest = audit.get("latest_start_kyiv", "невідомо")
    return (f"Прогноз сформовано зі зрізу {origin}. Найпізніший початок у цьому "
            f"знімку джерела: {latest}. Це не онлайн-оновлення; повноту "
            "наступних діб не підтверджено.")
