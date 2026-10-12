"""Stage 3 correction: select models on pre-holdout data, then evaluate honestly.

Keep run_data.py and run_regimes.py unchanged. All dates are Kyiv calendar days.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data import KYIV
from forecast import (
    ForecastError, evaluate_region, fit_probabilistic_model,
    get_region_series, predict_for_dates, read_and_validate_inputs,
    selected_probabilities, SELECTION_END, SELECTION_TARGET_END,
)
from regimes import RegimeDataError, HOLDOUT_END, HOLDOUT_START, TRAIN_END

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DAILY = OUT / "daily_alert_hours.csv"
THRESHOLDS = OUT / "regime_thresholds.csv"
AS_OF = date(2026, 10, 9)
FORECAST_START = date(2026, 10, 12)
FORECAST_END = date(2026, 10, 18)


def mean_brier(frame: pd.DataFrame, col: str) -> float:
    return float(frame[col].mean())


def main() -> int:
    try:
        daily, thresholds = read_and_validate_inputs(DAILY, THRESHOLDS)
        validation_rows, a_rows, b_rows, future_rows, choices, metrics = [], [], [], [], [], []
        for item in thresholds.itertuples(index=False):
            region = item.region
            series = get_region_series(daily, region, AS_OF)
            low, high = float(item.q33_hours), float(item.q75_hours)
            val, a, b, selected = evaluate_region(series, low, high)
            for frame, holder in ((val, validation_rows), (a, a_rows), (b, b_rows)):
                tagged = selected_probabilities(frame, selected)
                tagged.insert(0, "region", region)
                holder.append(tagged)

            # Future forecast is generated once from the Oct 9 snapshot.
            future_model = fit_probabilistic_model(series, low, high, AS_OF)
            target_dates = [FORECAST_START + timedelta(days=i) for i in range(7)]
            future = predict_for_dates(future_model, series, AS_OF, target_dates, low, high)
            future = future.rename(columns={c: f"model_{c}" for c in ("low", "medium", "high")})
            future = selected_probabilities(future, selected)
            future.insert(0, "region", region)
            future_rows.append(future)

            choices.append({
                "region": region,
                "selected_model": selected,
                "selection_end": SELECTION_TARGET_END.isoformat(),
                "selection_gap_brier_logistic": mean_brier(val, "model_brier"),
                "selection_gap_brier_baseline": mean_brier(val, "baseline_brier"),
                "selection_forecasts": len(val),
            })
            metrics.append({
                "region": region,
                "selected_model_from_validation": selected,
                "holdout_A_forecasts": len(a),
                "holdout_A_unique_dates": a["date"].nunique(),
                "holdout_A_logistic_brier": mean_brier(a, "model_brier"),
                "holdout_A_baseline_brier": mean_brier(a, "baseline_brier"),
                "holdout_A_selected_brier": mean_brier(selected_probabilities(a, selected), "selected_brier"),
                "holdout_B_forecasts": len(b),
                "holdout_B_unique_dates": b["date"].nunique(),
                "holdout_B_logistic_brier": mean_brier(b, "model_brier"),
                "holdout_B_baseline_brier": mean_brier(b, "baseline_brier"),
                "holdout_B_selected_brier": mean_brier(selected_probabilities(b, selected), "selected_brier"),
                "holdout_B_selected_skill_vs_baseline": (
                    1 - mean_brier(selected_probabilities(b, selected), "selected_brier") /
                    mean_brier(b, "baseline_brier")
                ) if mean_brier(b, "baseline_brier") > 0 else float("nan"),
            })
            print(f"{region:24s} | обрана: {selected:8s} | "
                  f"валідація B: logistic={mean_brier(val, 'model_brier'):.4f}, "
                  f"baseline={mean_brier(val, 'baseline_brier'):.4f} | "
                  f"тест B: selected={metrics[-1]['holdout_B_selected_brier']:.4f}")

        OUT.mkdir(parents=True, exist_ok=True)
        pd.concat(validation_rows, ignore_index=True).to_csv(
            OUT / "selection_validation.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(choices).to_csv(OUT / "model_selection.csv", index=False, encoding="utf-8-sig")
        a_all = pd.concat(a_rows, ignore_index=True)
        b_all = pd.concat(b_rows, ignore_index=True)
        a_all.to_csv(OUT / "backtest_predictions.csv", index=False, encoding="utf-8-sig")
        b_all.to_csv(OUT / "backtest_gap_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(metrics).to_csv(OUT / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(future_rows, ignore_index=True).to_csv(
            OUT / "week_2026-10-12_to_18.csv", index=False, encoding="utf-8-sig")
        manifest = {
            "created_at_kyiv": datetime.now(KYIV).isoformat(),
            "date_of_last_complete_observation": AS_OF.isoformat(),
            "forecast_target_dates": [FORECAST_START.isoformat(), FORECAST_END.isoformat()],
            "forecast_horizons": [3, 9],
            "model_selection_policy": "choose per region on pre-holdout validation gap_3_9; ties prefer baseline",
            "selection_threshold_reference_180days_through": SELECTION_END.isoformat(),
            "selection_target_range": ["2026-08-15", SELECTION_TARGET_END.isoformat()],
            "final_thresholds_through": TRAIN_END.isoformat(),
            "holdout_target_range": [HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat()],
            "holdout_A": "4 weekly issue dates, horizons 1-7, 28 unique dates per region",
            "holdout_B": "4 weekly issue dates, horizons 3-9, 26 unique dates per region",
            "model_refit": "logistic regression refitted at each weekly issue date, baseline recomputed from latest 28 days",
            "metric": "unscaled multiclass Brier = sum of three squared probability errors, lower is better",
            "daily_sha256": hashlib.sha256(DAILY.read_bytes()).hexdigest(),
            "thresholds_sha256": hashlib.sha256(THRESHOLDS.read_bytes()).hexdigest(),
            "limitations": [
                "The Sep12-Oct9 holdout was previously inspected in Stage 3; this is a corrected evaluation, not a fresh untouched holdout.",
                "A and B are overlapping experiments; don't pool them as independent targets.",
                "Selection validation uses earlier fixed thresholds fitted through Aug14; final test uses frozen Sep11 thresholds.",
                "CSV publication/retrospective edit times are not recorded: complete historical point-in-time availability is not proven.",
                "A dense calendar with zero recorded hours is not proof that zero alerts occurred.",
                "Naive-ended intervals are assumed 30 minutes and official yellow/red alarm levels aren't available.",
                "These are recorded-activity categories, not safety or actionable attack predictions.",
                "Oct12-18 is a forward forecast: its actual error is not yet known."
            ],
        }
        (OUT / "forecast_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n✅ Етап 3 виправлено. В outputs створено:")
        for f in ("model_selection.csv", "selection_validation.csv",
                  "forecast_metrics.csv", "backtest_predictions.csv",
                  "backtest_gap_predictions.csv", "week_2026-10-12_to_18.csv",
                  "forecast_manifest.json"):
            print("  " + f)
        print("⚠️ Обрану модель визначено за валідацією ДО фінального тесту.")
        print("⚠️ Прогноз низької активності НЕ означає безпечний день.")
        return 0
    except (ForecastError, RegimeDataError, ValueError, OSError, KeyError) as exc:
        print(f"❌ Помилка етапу 3: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
