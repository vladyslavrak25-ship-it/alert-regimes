"""Download the upstream Volunteer CSV and issue a pre-week seven-day forecast.

Run on Sunday evening in UTC, before next Monday 00:00 Europe/Kyiv.
Frozen model choices and thresholds stay unchanged; no historical test refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from data import KYIV, load_records, suggested_last_day
from forecast import ForecastError, load_thresholds
from safe_week import forecast_from_records, load_choices, frozen_fingerprint

ROOT = Path(__file__).resolve().parent
SOURCE_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
)
MAX_HORIZON = 9  # Historical protocol assessed horizons through 9, not farther.
MIN_CSV_BYTES = 100_000
MAX_CSV_BYTES = 60_000_000


def next_monday(now: datetime) -> date:
    """Next Monday strictly in the future (in the Kyiv civil calendar)."""
    local_today = now.astimezone(KYIV).date()
    days = (7 - local_today.weekday()) % 7
    return local_today + timedelta(days=days or 7)


def select_origin(safe_last_day: date, week_start: date, now: datetime) -> date:
    """Require a future forecast, source freshness and tested forecast horizons."""
    if now.astimezone(KYIV) >= datetime.combine(week_start, time.min, tzinfo=KYIV):
        raise ForecastError("Тиждень уже почався: не публікуємо заднім числом як прогноз.")
    origin = min(safe_last_day, week_start - timedelta(days=1))
    if origin >= week_start:
        raise ForecastError("Дата зрізу не може входити до прогнозного тижня.")
    if (week_start - origin).days + 6 > MAX_HORIZON:
        raise ForecastError(
            f"Джерело застаріле: останній обережно повний день {safe_last_day}. "
            f"Для тижня {week_start} потрібен зріз із горизонтом не далі {MAX_HORIZON}."
        )
    return origin


def download_source(to_path: Path, url: str = SOURCE_URL) -> str:
    """Download with bounded size into a temporary path, never touch pinned old CSV."""
    if url != SOURCE_URL:
        raise ValueError("Для автоматичного запуску дозволено лише зафіксоване GitHub-джерело.")
    request = Request(url, headers={"User-Agent": "Alert-Regimes-research/1.0"})
    count = 0
    digest = hashlib.sha256()
    with urlopen(request, timeout=60) as stream, to_path.open("wb") as out:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > MAX_CSV_BYTES:
                raise ValueError("Завантажений CSV завеликий — змінився контракт джерела.")
            digest.update(chunk)
            out.write(chunk)
    if count < MIN_CSV_BYTES:
        raise ValueError("Файл занадто малий: джерело може бути порожнім або недоступним.")
    with to_path.open("rb") as inp:
        header = inp.readline().decode("utf-8-sig").strip().replace('"', "")
    if not {"region", "started_at", "finished_at", "naive"}.issubset(set(header.split(","))):
        raise ValueError("Джерело змінило CSV-схему, прогноз зупинено.")
    return digest.hexdigest()


def ensure_no_duplicate_run(folder: Path, newest: Path) -> None:
    """Do not rewrite historical forecasts (including on reruns)."""
    if folder.exists():
        raise ForecastError(f"Прогноз цього тижня вже існує: {folder}. Не переписуємо архів.")
    # Latest must never be newer than the proposed target week.
    if newest.exists():
        try:
            previous = date.fromisoformat(json.loads(newest.read_text(encoding="utf-8"))["forecast_first_day"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            raise ForecastError("Пошкоджено previous auto-week manifest; відмовлено в оновленні.") from exc
        if previous >= date.fromisoformat(folder.name):
            raise ForecastError("Не публікуємо старішу версію замість новішого тижня.")


def run(root: Path, now: datetime, week_start: date | None = None) -> dict:
    root = Path(root)
    week_start = week_start or next_monday(now)
    if week_start.weekday() != 0:
        raise ForecastError("Потрібен початок у понеділок.")
    if now.astimezone(KYIV) >= datetime.combine(week_start, time.min, tzinfo=KYIV):
        raise ForecastError("Запізнілий запуск: тиждень уже почався.")
    history_folder = root / "analysis" / "auto_forecasts" / week_start.isoformat()
    latest_file = root / "outputs" / "auto_week_latest.csv"
    latest_manifest = root / "outputs" / "auto_week_latest_manifest.json"
    ensure_no_duplicate_run(history_folder, latest_manifest)
    thresholds_path = root / "outputs" / "regime_thresholds.csv"
    selection_path = root / "outputs" / "model_selection.csv"
    thresholds = load_thresholds(thresholds_path)
    choices = load_choices(selection_path)
    with tempfile.TemporaryDirectory(prefix="alert-regimes-fetch-") as tmp:
        downloaded = Path(tmp) / "volunteer_data_en.csv"
        source_sha = download_source(downloaded)
        records, audit = load_records(downloaded)
        last_safe = suggested_last_day(records, today=now.astimezone(KYIV).date())
        origin = select_origin(last_safe, week_start, now)
        forecast, _ = forecast_from_records(records, thresholds, choices, week_start, as_of=origin)
        if len(forecast) != 49 or forecast["date"].nunique() != 7:
            raise ForecastError("Очікується 49 прогнозів (7 областей × 7 днів).")
        source_latest_start = records["started_at"].max().to_pydatetime().astimezone(KYIV).isoformat()
        sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        meta = {
            "forecast_first_day": week_start.isoformat(),
            "forecast_last_day": (week_start + timedelta(days=6)).isoformat(),
            "last_training_day": origin.isoformat(),
            "forbidden_from_kyiv": f"{origin + timedelta(days=1)}T00:00 Europe/Kyiv",
            "horizons": list(range((week_start-origin).days, (week_start-origin).days + 7)),
            "created_at_kyiv": now.astimezone(KYIV).isoformat(),
            "source_url": SOURCE_URL,
            "source_sha256": source_sha,
            "source_latest_start_kyiv": source_latest_start,
            "source_rows_valid": int(len(records)),
            "invalid_rows_removed": int(audit["invalid_rows_removed"]),
            "accepted_past_sha256": frozen_fingerprint(records, origin + timedelta(days=1)),
            "thresholds_sha256": sha(thresholds_path),
            "model_selection_sha256": sha(selection_path),
            "selection_policy": "Frozen Sep-11-2026 validation, 5 baseline and 2 logistic, no re-selection",
            "availability_caveat": "Snapshot captured at recorded time; per-event publication timestamps unavailable. Latest-start-minus-one is a conservative proxy, NOT proof of full daily capture.",
            "disclaimer": "Statistical prediction of relative recorded alert-duration categories; not a safety warning.",
        }
        csv_bytes = forecast.to_csv(index=False).encode("utf-8-sig")
    # Only publish after successful data checks AND completed prediction.
    history_folder.mkdir(parents=True, exist_ok=False)
    (history_folder / "forecast_7days.csv").write_bytes(csv_bytes)
    (history_folder / "run_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_bytes(csv_bytes)
    latest_manifest.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe scheduled pre-week update from upstream Volunteer CSV")
    parser.add_argument("--start", type=date.fromisoformat, help="Next Monday YYYY-MM-DD; default auto")
    args = parser.parse_args()
    try:
        result = run(ROOT, datetime.now(timezone.utc), args.start)
    except (OSError, ValueError, ForecastError, KeyError) as exc:
        print(f"❌ Автоматичний прогноз не опубліковано: {exc}")
        return 1
    print("✅ Опубліковано тиждень", result["forecast_first_day"], "—", result["forecast_last_day"])
    print("Зріз:", result["last_training_day"], "Горизонти:", result["horizons"])
    print("Попередні фінальні файли Stage 3 не змінено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
