"""Run the first data-processing step from the PyCharm terminal."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from data import DataQualityError, KYIV, daily_hours, load_records, suggested_last_day

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "volunteer_data_en.csv"
OUTPUT_DIR = ROOT / "outputs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Підрахунок годин тривог за днями")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Шлях до Volunteer CSV")
    parser.add_argument("--through", type=date.fromisoformat, help="Остання дата YYYY-MM-DD")
    args = parser.parse_args()

    try:
        records, audit = load_records(args.csv)
        last_start_local = records["started_at"].max().to_pydatetime().astimezone(KYIV)
        today = datetime.now(KYIV).date()
        safe_day = suggested_last_day(records, today=today)
        last_day = args.through or safe_day
        if last_day > safe_day:
            raise DataQualityError(
                f"Дата {last_day} пізніша за обережну межу {safe_day}. "
                "Останній день із записами може бути неповним."
            )
        result = daily_hours(records, through=last_day)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        result_path = OUTPUT_DIR / "daily_alert_hours.csv"
        audit_path = OUTPUT_DIR / "data_audit.json"
        result.to_csv(result_path, index=False, encoding="utf-8-sig")
        audit.update({
            "data_file": str(args.csv),
            "input_sha256": sha256_file(args.csv),
            "report_created_at_kyiv": datetime.now(KYIV).isoformat(),
            "latest_start_kyiv": last_start_local.isoformat(),
            "last_included_day": last_day.isoformat(),
            "number_of_regions": int(result["region"].nunique()),
            "number_of_daily_rows": int(len(result)),
            "days_without_source_intervals": int((~result["has_source_interval"]).sum()),
            "note": (
                "0 recorded hours means no matching interval in this CSV, "
                "not proof that no alert occurred. The dataset can be revised retrospectively."
            ),
        })
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n✅ Обробку завершено.")
        print(f"Рядків у вихідному CSV: {audit['input_rows']:,}")
        print(f"Відхилено некоректних: {audit['invalid_rows_removed']}")
        print(f"Видалено точних дублікатів: {audit['exact_duplicate_rows_removed']}")
        print(f"Останній початок у джерелі (Київ): {last_start_local:%Y-%m-%d %H:%M:%S}")
        print(f"Остання врахована дата: {last_day}")
        print(f"Регіонів у результаті: {audit['number_of_regions']}")
        print(f"Збережено: {result_path}")
        print(f"Звіт перевірки: {audit_path}")
        print("⚠️ 0 записаних годин НЕ доводить, що в області не було тривоги.")
        if (today - last_start_local.date()).days >= 2:
            print("⚠️ Увага: найновіший початок у CSV старший за два календарні дні.")
        return 0
    except (DataQualityError, ValueError, OSError) as exc:
        print(f"❌ Помилка даних: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
