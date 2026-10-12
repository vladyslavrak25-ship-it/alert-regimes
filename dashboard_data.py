"""Load and verify precomputed Alert Regimes artifacts for the Streamlit UI.

The dashboard does NOT train models, select models or modify saved forecasts.
All dates use the Kyiv civil calendar as already prepared by Stage 1.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


class DashboardDataError(ValueError):
    """Readable message for missing or inconsistent artifacts."""


REGION_UK = {
    "Chernihivska oblast": "Чернігівська область",
    "Kharkivska oblast": "Харківська область",
    "Sumska oblast": "Сумська область",
    "Zaporizka oblast": "Запорізька область",
    "Dnipropetrovska oblast": "Дніпропетровська область",
    "Poltavska oblast": "Полтавська область",
    "Mykolaivska oblast": "Миколаївська область",
}
REGIME_UK = {"low": "Низька", "medium": "Помірна", "high": "Висока"}
MODEL_UK = {"baseline": "Baseline", "logistic": "Logistic Regression"}

FILES = {
    "daily": "daily_alert_hours.csv",
    "thresholds": "regime_thresholds.csv",
    "selection": "model_selection.csv",
    "metrics": "forecast_metrics.csv",
    "future": "week_2026-10-12_to_18.csv",
    "backtest_a": "backtest_predictions.csv",
    "backtest_b": "backtest_gap_predictions.csv",
    "manifest": "forecast_manifest.json",
    "audit": "data_audit.json",
}
SELECTION_FIELDS = [
    "region", "selected_model", "selection_end",
    "selection_gap_brier_logistic", "selection_gap_brier_baseline",
    "selection_forecasts",
]


def _csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"Не знайдено {path.name}. Спочатку запусти відповідний етап Python.")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        raise DashboardDataError(f"Не вдалося прочитати {path.name}: {exc}") from exc
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DashboardDataError(f"У {path.name} відсутні стовпці: {', '.join(missing)}")
    if frame.empty:
        raise DashboardDataError(f"Файл {path.name} порожній.")
    return frame


def load_model_selection(path: Path) -> pd.DataFrame:
    """Support both normal CSV exports and the six-column headerless export."""
    if not path.is_file():
        raise DashboardDataError("Немає model_selection.csv. Запусти run_forecast.py.")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not set(SELECTION_FIELDS).issubset(frame.columns):
            frame = pd.read_csv(path, header=None, names=SELECTION_FIELDS, encoding="utf-8-sig")
    except Exception as exc:
        raise DashboardDataError(f"Не вдалося прочитати model_selection.csv: {exc}") from exc
    if frame.empty or frame["region"].duplicated().any():
        raise DashboardDataError("Вибір моделей порожній або містить повтори областей.")
    if not frame["selected_model"].isin(MODEL_UK).all():
        raise DashboardDataError("model_selection.csv містить невідомий тип моделі.")
    for name in ("selection_gap_brier_logistic", "selection_gap_brier_baseline"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        if not np.isfinite(frame[name]).all():
            raise DashboardDataError(f"Некоректні значення {name} в model_selection.csv.")
    return frame


def _parse_dates(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> pd.DataFrame:
    for col in columns:
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
        if frame[col].isna().any():
            raise DashboardDataError(f"У {source} є неправильні дати в полі {col}.")
    return frame


def check_probabilities(frame: pd.DataFrame, prefix: str = "selected") -> None:
    columns = [f"{prefix}_{category}" for category in ("low", "medium", "high")]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DashboardDataError(f"Відсутні ймовірності: {', '.join(missing)}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -1e-8).any() or (values > 1 + 1e-8).any():
        raise DashboardDataError("Прогноз містить некоректні ймовірності.")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise DashboardDataError("Сума ймовірностей низької, помірної та високої активності не дорівнює 1.")


def _check_unique_targets(frame: pd.DataFrame, source: str) -> None:
    if frame.duplicated(["region", "date"]).any():
        raise DashboardDataError(f"У {source} знайдені повторні прогнози тієї самої дати для області.")
    if (frame["date"] <= frame["origin"]).any():
        raise DashboardDataError(f"У {source} прогнозна дата не пізніша за дату зрізу.")
    difference = (frame["date"] - frame["origin"]).dt.days
    if (difference != pd.to_numeric(frame["horizon"], errors="coerce")).any():
        raise DashboardDataError(f"У {source} неправильно обчислено горизонт прогнозу.")


def load_outputs(folder: Path) -> dict:
    folder = Path(folder)
    required_common = {"region", "date"}
    daily = _parse_dates(_csv(folder / FILES["daily"], required_common | {
        "recorded_hours", "naive_added_hours"
    }), ("date",), FILES["daily"])
    if daily.duplicated(["region", "date"]).any():
        raise DashboardDataError("Добова таблиця містить дублікати region + date.")
    hours = pd.to_numeric(daily["recorded_hours"], errors="coerce")
    naive = pd.to_numeric(daily["naive_added_hours"], errors="coerce")
    if not np.isfinite(hours).all() or not np.isfinite(naive).all() or (hours < 0).any() or (hours > 24.00001).any() or (naive < 0).any():
        raise DashboardDataError("Некоректна добова тривалість (має бути від 0 до 24 годин).")

    thresholds = _csv(folder / FILES["thresholds"], {"region", "q33_hours", "q75_hours"})
    selection = load_model_selection(folder / FILES["selection"])
    metrics = _csv(folder / FILES["metrics"], {
        "region", "holdout_A_logistic_brier", "holdout_A_baseline_brier",
        "holdout_B_logistic_brier", "holdout_B_baseline_brier",
        "holdout_B_selected_brier",
    })
    future = _parse_dates(_csv(folder / FILES["future"], required_common | {
        "origin", "horizon", "selected_model", "selected_low", "selected_medium", "selected_high"
    }), ("date", "origin"), FILES["future"])
    backtests = {}
    for name, key in (("next", "backtest_a"), ("gap", "backtest_b")):
        frame = _parse_dates(_csv(folder / FILES[key], required_common | {
            "origin", "horizon", "actual_regime", "model_brier", "baseline_brier",
            "selected_brier", "selected_low", "selected_medium", "selected_high",
        }), ("date", "origin"), FILES[key])
        _check_unique_targets(frame, FILES[key])
        check_probabilities(frame)
        if not frame["actual_regime"].isin(REGIME_UK).all():
            raise DashboardDataError(f"Некоректна фактична категорія у {FILES[key]}.")
        backtests[name] = frame
    _check_unique_targets(future, FILES["future"])
    check_probabilities(future)
    if not future["selected_model"].isin(MODEL_UK).all():
        raise DashboardDataError("Невідомі моделі в прогнозі.")
    selection_map = selection.set_index("region")["selected_model"].to_dict()
    if set(future["region"]) != set(selection_map):
        raise DashboardDataError("Набір областей у прогнозі не збігається з вибором моделей.")
    if any(selection_map[r] != m for r, m in zip(future["region"], future["selected_model"])):
        raise DashboardDataError("Обрана модель у прогнозі не збігається з model_selection.csv.")
    for table_name, table in (("thresholds", thresholds), ("metrics", metrics)):
        if set(table["region"]) != set(selection_map):
            raise DashboardDataError(f"Області в {table_name} не збігаються з прогнозом.")
    try:
        manifest = json.loads((folder / FILES["manifest"]).read_text(encoding="utf-8"))
        audit = json.loads((folder / FILES["audit"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Немає або пошкоджено звіт аудиту/прогнозу: {exc}") from exc
    cutoff = pd.Timestamp(manifest["date_of_last_complete_observation"])
    # Fail closed on mixed snapshots: refreshing Stage 1 alone must not silently
    # relabel an old Stage 3 forecast as if trained on the new source.
    expected_digest = manifest.get("daily_sha256")
    if expected_digest:
        observed_digest = hashlib.sha256((folder / FILES["daily"]).read_bytes()).hexdigest()
        if observed_digest != expected_digest:
            raise DashboardDataError(
                "daily_alert_hours.csv не збігається зі знімком, з якого сформовано прогноз. "
                "Не змішуй результати різних запусків; збережи поточні outputs окремо."
            )
    if audit.get("last_included_day") != cutoff.date().isoformat():
        raise DashboardDataError("Дата знімка історії не збігається з датою зрізу прогнозу.")
    if (future["origin"] != cutoff).any():
        raise DashboardDataError("Дата зрізу в прогнозі не збігається з forecast_manifest.json.")
    # This dashboard represents the fixed 12–18 October forecast, NOT the live future.
    dates = pd.date_range("2026-10-12", "2026-10-18", freq="D")
    for region in selection_map:
        region_dates = future.loc[future["region"] == region, "date"].sort_values()
        if list(region_dates) != list(dates):
            raise DashboardDataError(f"Для {region} немає рівно семи дат 12–18 жовтня.")
    return {
        "daily": daily, "thresholds": thresholds, "selection": selection,
        "metrics": metrics, "future": future, "backtests": backtests,
        "manifest": manifest, "audit": audit,
    }


def classify_daily(hours: pd.Series, q33: float, q75: float) -> pd.Series:
    if q33 >= q75:
        raise DashboardDataError("Пороги активності збігаються або переплутані.")
    values = pd.to_numeric(hours, errors="coerce")
    if values.isna().any():
        raise DashboardDataError("Не вдалося класифікувати частину днів.")
    return pd.Series(np.where(values <= q33, "low", np.where(values <= q75, "medium", "high")), index=hours.index)
