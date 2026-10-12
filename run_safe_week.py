"""Create a SEPARATE seven-day forecast with strict Kyiv-calendar cutoff.

Example:
  python run_safe_week.py --csv "data/volunteer_data_en_candidate.csv" --start 2026-10-12
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from data import DataQualityError, KYIV, load_records, suggested_last_day
from forecast import ForecastError, load_thresholds
from safe_week import forecast_from_records, load_choices, save_new_run

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Прогноз лише з минулих спостережень; без перезапису Stage 3.")
    parser.add_argument("--csv", type=Path, required=True, help="Новий CSV; вихідний файл НЕ змінюється")
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="Перший з 7 днів YYYY-MM-DD")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "outputs" / "regime_thresholds.csv")
    parser.add_argument("--choices", type=Path, default=ROOT / "outputs" / "model_selection.csv")
    parser.add_argument("--outdir", type=Path, help="Нова окрема папка, яку ще не створено")
    args = parser.parse_args(argv)
    try:
        records, audit = load_records(args.csv)
        safe = suggested_last_day(records, today=datetime.now(KYIV).date())
        as_of = args.start - timedelta(days=1)
        if as_of > safe:
            raise ForecastError(
                f"Не підтверджено навіть консервативної повноти дня {as_of}; "
                f"поточна обережна межа {safe}. Завантаж новіше джерело."
            )
        print(f"Джерело: {len(records):,} валідних інтервалів, {audit['invalid_rows_removed']} невалідних.")
        print(f"Дата прогнозу: {args.start}…{args.start + timedelta(days=6)}; навчальна межа: {as_of} включно.")
        print("ВАЖЛИВО: наявність записів не доводить, що вони були опубліковані до початку прогнозу.")
        thresholds = load_thresholds(args.thresholds)
        choices = load_choices(args.choices)
        forecast, accepted_daily = forecast_from_records(records, thresholds, choices, args.start)
        path = args.outdir or ROOT / "analysis" / f"safe_forecast_{args.start.isoformat()}"
        paths = save_new_run(args.csv, args.start, forecast, accepted_daily,
                             records, path, args.thresholds, args.choices)
        print(f"✅ {len(forecast)} прогнозів (7 областей × 7 днів), горизонти 1–7.")
        for p in paths:
            print(f"  {p}")
        print("Старі файли outputs і застосунок Streamlit не змінено.")
        return 0
    except (DataQualityError, ForecastError, OSError, ValueError, KeyError) as exc:
        print(f"❌ {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
