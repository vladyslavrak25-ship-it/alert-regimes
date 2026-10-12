"""Stage 2 runner: freeze regime thresholds before our 28-day holdout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from data import KYIV
from regimes import (
    CANDIDATE_REGIONS,
    HOLDOUT_END,
    HOLDOUT_START,
    TRAIN_END,
    TRAIN_START,
    RegimeDataError,
    build_training_regimes,
    read_daily_table,
)

ROOT = Path(__file__).resolve().parent
DAILY_PATH = ROOT / "outputs" / "daily_alert_hours.csv"
OUTPUT_DIR = ROOT / "outputs"


def main() -> int:
    try:
        daily = read_daily_table(DAILY_PATH)
        thresholds, labels, report = build_training_regimes(daily)
        OUTPUT_DIR.mkdir(exist_ok=True)
        thresholds.to_csv(OUTPUT_DIR / "regime_thresholds.csv", index=False, encoding="utf-8-sig")
        labels.to_csv(OUTPUT_DIR / "train_regimes.csv", index=False, encoding="utf-8-sig")
        report.to_csv(OUTPUT_DIR / "region_readiness.csv", index=False, encoding="utf-8-sig")
        manifest = {
            "created_at_kyiv": datetime.now(KYIV).isoformat(),
            "daily_csv_sha256": hashlib.sha256(DAILY_PATH.read_bytes()).hexdigest(),
            "training_period_inclusive": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
            "reserved_holdout_not_used_for_thresholds": [
                HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()
            ],
            "candidate_regions": list(CANDIDATE_REGIONS),
            "accepted_regions": thresholds["region"].tolist(),
            "definition": (
                "low: recorded_hours <= training q33; "
                "medium: training q33 < recorded_hours <= training q75; "
                "high: recorded_hours > training q75"
            ),
            "warning": (
                "These are relative recorded-activity labels, not official alarm colours "
                "or a guarantee of safety. No-alert rows may also reflect missing source data."
            ),
        }
        (OUTPUT_DIR / "regime_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n✅ Етап 2: категорії підготовлено.")
        print(f"Навчальна історія: {TRAIN_START} — {TRAIN_END}")
        print(f"Недоторканий тестовий період: {HOLDOUT_START} — {HOLDOUT_END}")
        print(f"Областей, які пройшли перевірку: {len(thresholds)} із {len(report)}")
        print("\nОбласть                 Пороги (год)       Класи (низ./пом./вис.)")
        for row in thresholds.itertuples(index=False):
            counts = report.loc[report["region"] == row.region].iloc[0]
            print(
                f"{row.region:24s} {row.q33_hours:5.2f} / {row.q75_hours:5.2f}      "
                f"{counts.low_days:3d} / {counts.medium_days:3d} / {counts.high_days:3d}"
            )
        for row in report.itertuples(index=False):
            if not row.suitable:
                print(f"⚠️ Виключено {row.region}: {row.reason}")
        print("\nФайли створено в папці outputs:")
        for file in ("regime_thresholds.csv", "train_regimes.csv", "region_readiness.csv", "regime_manifest.json"):
            print(f"  - {file}")
        print("⚠️ Тестові дні не використовувалися для підбору порогів.")
        return 0
    except (RegimeDataError, ValueError, OSError) as exc:
        print(f"❌ Помилка на етапі категорій: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
