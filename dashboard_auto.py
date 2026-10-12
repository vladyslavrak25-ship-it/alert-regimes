"""Read optional weekly automatic forecasts without changing the pinned historical results."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


class AutoForecastError(ValueError):
    pass


def load_latest_week(output_folder: Path, expected_models: dict[str, str]):
    """Return (forecast, manifest) or None if no automatic run exists yet."""
    folder = Path(output_folder)
    csv = folder / "auto_week_latest.csv"
    manifest_path = folder / "auto_week_latest_manifest.json"
    if not csv.exists() and not manifest_path.exists():
        return None
    if not csv.exists() or not manifest_path.exists():
        raise AutoForecastError("Автоматичний прогноз неповний: потрібні CSV та manifest разом.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        future = pd.read_csv(csv, encoding="utf-8-sig")
        start = date.fromisoformat(manifest["forecast_first_day"])
        end = date.fromisoformat(manifest["forecast_last_day"])
        origin = date.fromisoformat(manifest["last_training_day"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AutoForecastError("Некоректні файли автоматичного прогнозу.") from exc
    if start.weekday() != 0 or end != start + timedelta(days=6) or origin >= start:
        raise AutoForecastError("Некоректні часові межі автоматичного прогнозу.")
    required = {"region", "date", "origin", "horizon", "selected_model", "selected_low", "selected_medium", "selected_high"}
    if not required.issubset(future.columns) or len(future) != 49:
        raise AutoForecastError("Прогноз повинен мати 49 рядків і всі обов'язкові стовпці.")
    future["date"] = pd.to_datetime(future["date"], errors="coerce")
    future["origin"] = pd.to_datetime(future["origin"], errors="coerce")
    if future[["date", "origin"]].isna().any().any() or future.duplicated(["region", "date"]).any():
        raise AutoForecastError("Некоректні або повторні дати автоматичного прогнозу.")
    expected_dates = set(pd.date_range(start, end, freq="D"))
    if set(future["region"]) != set(expected_models):
        raise AutoForecastError("Набір областей не збігається з обраними моделями.")
    for region, group in future.groupby("region"):
        if set(group["date"]) != expected_dates:
            raise AutoForecastError(f"Неповний тиждень для {region}.")
        if not (group["selected_model"] == expected_models[region]).all():
            raise AutoForecastError(f"Несумісна обрана модель для {region}.")
    if not (future["origin"] == pd.Timestamp(origin)).all():
        raise AutoForecastError("Дата зрізу не збігається з manifest.")
    if not ((future["date"] - future["origin"]).dt.days == pd.to_numeric(future["horizon"], errors="coerce")).all():
        raise AutoForecastError("Некоректний горизонт прогнозу.")
    p = future[["selected_low", "selected_medium", "selected_high"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise AutoForecastError("Некоректні ймовірності прогнозу.")
    return future, manifest
