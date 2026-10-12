"""Strict data-leakage regression tests for a generic seven-day forecast."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from data import KYIV, clean_records
from forecast import ForecastError
from safe_week import (
    forecast_from_records, frozen_fingerprint, frozen_past, load_choices,
    midnight_utc, save_new_run, seven_dates,
)


def raw_records(last_day="2026-10-13"):
    days = pd.date_range("2025-04-01", last_day, freq="D")
    raw = []
    for i, day in enumerate(days):
        # Kyiv 10 AM - known interval; enough data to fit the 365-day model.
        local = day.tz_localize(KYIV) + pd.Timedelta(hours=10)
        span = [1.2, 4.0, 8.1][i % 3]
        raw.append({
            "region": "Chernihivska oblast", "started_at": local.isoformat(),
            "finished_at": (local + pd.Timedelta(hours=span)).isoformat(), "naive": "false"
        })
    return pd.DataFrame(raw)


def cleaned(raw):
    return clean_records(raw)[0]


def metadata(model="logistic"):
    t = pd.DataFrame([{
        "region": "Chernihivska oblast", "q33_hours": 2.5,
        "q75_hours": 6.0, "fit_start": "2026-03-16", "fit_end": "2026-09-11"
    }])
    c = pd.DataFrame([{
        "region": "Chernihivska oblast", "selected_model": model,
        "selection_end": "2026-09-11"
    }])
    return t, c


def test_kyiv_midnight_boundary_not_utc_midnight():
    assert pd.Timestamp(midnight_utc(date(2026, 10, 12))) == pd.Timestamp("2026-10-11T21:00:00Z")


def test_late_and_crossing_interval_capped_exactly():
    raw = pd.DataFrame([
        {"region": "Chernihivska oblast", "started_at": "2026-10-11T20:30:00Z",
         "finished_at": "2026-10-11T23:59:00Z", "naive": "false"},
        {"region": "Chernihivska oblast", "started_at": "2026-10-11T21:00:00Z",
         "finished_at": "2026-10-11T23:00:00Z", "naive": "false"},
    ])
    f = frozen_past(cleaned(raw), date(2026, 10, 12))
    assert len(f) == 1
    assert f.iloc[0]["finished_at"] == pd.Timestamp("2026-10-11T21:00:00Z")
    assert (f.iloc[0]["finished_at"] - f.iloc[0]["started_at"]).total_seconds() == 1800


def test_all_future_changes_leave_accepted_past_identical():
    r1 = raw_records()
    r2 = r1.copy()
    future = pd.to_datetime(r2["started_at"], utc=True).dt.tz_convert(KYIV).dt.date >= date(2026, 10, 12)
    r2.loc[future, "finished_at"] = (
        pd.to_datetime(r2.loc[future, "started_at"], utc=True) + pd.Timedelta(hours=14)
    ).astype(str).to_numpy()
    r2 = pd.concat([r2, pd.DataFrame([{
        "region": "Chernihivska oblast", "started_at": "2026-10-12T04:00:00Z",
        "finished_at": "2026-10-12T13:00:00Z", "naive": "true"
    }])], ignore_index=True)
    c1, c2 = cleaned(r1), cleaned(r2)
    pd.testing.assert_frame_equal(frozen_past(c1, date(2026, 10, 12)).reset_index(drop=True),
                                  frozen_past(c2, date(2026, 10, 12)).reset_index(drop=True))
    assert frozen_fingerprint(c1, date(2026, 10, 12)) == frozen_fingerprint(c2, date(2026, 10, 12))


@pytest.mark.parametrize("model", ["baseline", "logistic"])
def test_real_model_probabilities_do_not_change_when_future_records_change(model):
    r1 = raw_records()
    r2 = pd.concat([r1, pd.DataFrame([{
        "region": "Chernihivska oblast", "started_at": "2026-10-12T01:00:00Z",
        "finished_at": "2026-10-12T17:00:00Z", "naive": "true"
    }])], ignore_index=True)
    t, c = metadata(model)
    a, daily_a = forecast_from_records(cleaned(r1), t, c, date(2026, 10, 12))
    b, daily_b = forecast_from_records(cleaned(r2), t, c, date(2026, 10, 12))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(daily_a, daily_b)
    assert a["horizon"].tolist() == list(range(1, 8))
    assert all(a["origin"] == "2026-10-11")
    assert np.allclose(a[["selected_low", "selected_medium", "selected_high"]].sum(axis=1), 1)
    assert daily_a["date"].max() == "2026-10-11"
    if model == "baseline":
        assert a["selected_high"].nunique() == 1


def test_any_seven_consecutive_future_days_not_just_oct12():
    r = cleaned(raw_records(last_day="2026-10-20"))
    t, c = metadata("baseline")
    a, _ = forecast_from_records(r, t, c, date(2026, 10, 19))
    assert a["date"].tolist() == [str(d.date()) for d in pd.date_range("2026-10-19", periods=7)]
    assert a["origin"].unique().tolist() == ["2026-10-18"]


def test_thresholds_and_model_choice_are_frozen():
    r = cleaned(raw_records())
    t, c = metadata("baseline")
    bad_t = t.copy()
    bad_t["fit_end"] = "2026-10-12"
    with pytest.raises(ForecastError, match="11.09.2026"):
        forecast_from_records(r, bad_t, c, date(2026, 10, 12))
    bad_c = c.copy()
    bad_c["selection_end"] = "2026-10-12"
    with pytest.raises(ForecastError, match="зафіксований"):
        forecast_from_records(r, t, bad_c, date(2026, 10, 12))
    with pytest.raises(ForecastError):
        seven_dates(date(2026, 9, 11))


def test_support_headerless_selection(tmp_path):
    path = tmp_path / "models.csv"
    path.write_text("Chernihivska oblast,baseline,2026-09-11,0.55,0.44,26\n", encoding="utf-8")
    f = load_choices(path)
    assert f.iloc[0]["selected_model"] == "baseline"


def test_result_directory_is_immutable(tmp_path):
    raw = cleaned(raw_records())
    t, c = metadata("baseline")
    f, d = forecast_from_records(raw, t, c, date(2026, 10, 12))
    csv_path, tfile, cfile = (tmp_path / n for n in ("data.csv", "thresholds.csv", "models.csv"))
    csv_path.write_text("immutable source", encoding="utf-8")
    t.to_csv(tfile, index=False)
    c.to_csv(cfile, index=False)
    dest = tmp_path / "new_run"
    paths = save_new_run(csv_path, date(2026, 10, 12), f, d, raw, dest, tfile, cfile)
    saved = [p.read_bytes() for p in paths]
    with pytest.raises(ForecastError, match="вже існує"):
        save_new_run(csv_path, date(2026, 10, 12), f, d, raw, dest, tfile, cfile)
    assert [p.read_bytes() for p in paths] == saved
