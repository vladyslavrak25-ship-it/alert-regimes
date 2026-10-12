from datetime import date, datetime, timezone
from pathlib import Path
import json

import pandas as pd
import pytest

from data import KYIV
from forecast import ForecastError
from weekly_autorun import next_monday, select_origin, ensure_no_duplicate_run, SOURCE_URL
from safe_week import forecast_from_records


def test_next_week_is_strictly_future():
    sunday = datetime(2026, 10, 18, 18, 0, tzinfo=timezone.utc)
    assert next_monday(sunday) == date(2026, 10, 19)
    monday = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
    assert next_monday(monday) == date(2026, 10, 26)


def test_sunday_cutoff_uses_saturday_and_days_2_through_8():
    now = datetime(2026, 10, 18, 18, 0, tzinfo=timezone.utc)
    assert select_origin(date(2026, 10, 17), date(2026, 10, 19), now) == date(2026, 10, 17)
    assert [(date(2026, 10, 19) - date(2026, 10, 17)).days + i for i in range(7)] == list(range(2, 9))


def test_stale_data_refuses_forecast():
    now = datetime(2026, 10, 18, 18, tzinfo=timezone.utc)
    with pytest.raises(ForecastError, match="застаріле"):
        select_origin(date(2026, 10, 14), date(2026, 10, 19), now)


def test_after_week_start_refuses_backdated_forecast():
    with pytest.raises(ForecastError, match="уже почався"):
        select_origin(date(2026, 10, 18), date(2026, 10, 19),
                      datetime(2026, 10, 19, 7, tzinfo=timezone.utc))


def test_existing_history_is_not_overwritten(tmp_path):
    folder = tmp_path / "2026-10-19"
    folder.mkdir()
    with pytest.raises(ForecastError, match="вже існує"):
        ensure_no_duplicate_run(folder, tmp_path / "manifest.json")


def test_latest_cannot_regress(tmp_path):
    manifest = tmp_path / "latest.json"
    manifest.write_text(json.dumps({"forecast_first_day":"2026-10-26"}))
    with pytest.raises(ForecastError, match="старішу"):
        ensure_no_duplicate_run(tmp_path / "2026-10-19", manifest)


def test_source_is_exact_raw_github_file():
    assert SOURCE_URL.endswith("/main/datasets/volunteer_data_en.csv")
    assert "raw.githubusercontent.com/Vadimkin/" in SOURCE_URL

@pytest.mark.parametrize("choice", ["baseline", "logistic"])
def test_extra_day_gap_has_no_future_data_leakage(choice):
    from data import clean_records
    # Enough training history for both actual algorithms; hypothetical Oct 19 week.
    records = []
    for i, day in enumerate(pd.date_range("2025-04-01", "2026-10-18", freq="D")):
        start = day.tz_localize(KYIV) + pd.Timedelta(hours=10)
        duration = [1.2, 4, 8.1][i % 3]
        records.append({"region": "Chernihivska oblast", "started_at": start.isoformat(),
                        "finished_at": (start + pd.Timedelta(hours=duration)).isoformat(), "naive": "false"})
    base = pd.DataFrame(records)
    extra = pd.DataFrame([{
        "region": "Chernihivska oblast",
        "started_at": "2026-10-18T08:00:00+03:00",  # Forbidden Sunday.
        "finished_at": "2026-10-18T21:00:00+03:00", "naive": "true",
    }])
    thresholds = pd.DataFrame([{
        "region": "Chernihivska oblast", "q33_hours": 2.5, "q75_hours": 6.0,
        "fit_start": "2026-03-16", "fit_end": "2026-09-11",
    }])
    selection = pd.DataFrame([{
        "region": "Chernihivska oblast", "selected_model": choice,
        "selection_end": "2026-09-11",
    }])
    a, daily_a = forecast_from_records(clean_records(base)[0], thresholds, selection,
                                        date(2026, 10, 19), as_of=date(2026, 10, 17))
    b, daily_b = forecast_from_records(clean_records(pd.concat([base, extra], ignore_index=True))[0],
                                        thresholds, selection, date(2026, 10, 19),
                                        as_of=date(2026, 10, 17))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(daily_a, daily_b)
    assert a["horizon"].tolist() == list(range(2, 9))
    assert daily_a["date"].max() == "2026-10-17"
