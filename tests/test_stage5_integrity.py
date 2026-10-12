"""Stage 5: model identity, leak-free frozen windows, full history and day clicks."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_analysis import (
    regional_category_counts, selected_day_forecast, subset_all_regions,
    verify_frozen_windows, verify_model_choices,
)
from dashboard_data import DashboardDataError, load_outputs
from forecast import selection_thresholds, get_region_series
from regimes import build_training_regimes, TRAIN_START, TRAIN_END, CANDIDATE_REGIONS

OUT = Path(__file__).resolve().parents[1] / "outputs"


@pytest.fixture(scope="module")
def sample():
    return load_outputs(OUT)


def test_full_historical_table_is_not_limited_to_training_window(sample):
    regions = list(sample["selection"]["region"])
    frame = subset_all_regions(sample["daily"], regions,
                               date(2022, 2, 25), date(2026, 10, 9), date(2026, 10, 9))
    assert len(frame) > len(regions) * 180
    assert frame["date"].min().date().year == 2022
    assert frame["date"].max().date() == date(2026, 10, 9)
    assert frame.groupby("region")["date"].nunique().min() > 1600


def test_regime_counts_sum_to_actual_calendar_rows(sample):
    regions = list(sample["selection"]["region"])
    frame = subset_all_regions(sample["daily"], regions,
                               date(2026, 3, 16), date(2026, 10, 9), date(2026, 10, 9))
    table = regional_category_counts(frame, sample["thresholds"], regions)
    assert int(table[["Низька", "Помірна", "Висока"]].to_numpy().sum()) == len(frame)
    assert (table[["Низька", "Помірна", "Висока"]].sum(axis=1)
            == table["Усього днів"]).all()
    assert (table["Усього днів"] == 208).all()  # 16 Mar–9 Oct, inclusive


@pytest.mark.parametrize("shift", range(7))
def test_day_click_returns_exactly_seven_correct_dates(sample, shift):
    target = date(2026, 10, 12 + shift)
    regions = list(sample["selection"]["region"])
    view = selected_day_forecast(sample["future"], target, regions)
    assert len(view) == 7
    assert set(view["region"]) == set(regions)
    assert (pd.to_datetime(view["date"]).dt.date == target).all()
    assert (pd.to_numeric(view["horizon"]) == 3 + shift).all()
    assert np.allclose(view[["Низька", "Помірна", "Висока"]].sum(axis=1), 100)


def test_each_region_uses_only_preselected_model(sample):
    results = verify_model_choices(sample["future"], sample["selection"])
    assert results == {"baseline": 5, "logistic": 2}


def test_baseline_same_probability_for_week_and_no_artificial_day_fluctuation(sample):
    for region in sample["selection"].loc[
        sample["selection"]["selected_model"] == "baseline", "region"
    ]:
        frame = sample["future"].loc[sample["future"]["region"] == region]
        probs = frame[["selected_low", "selected_medium", "selected_high"]].to_numpy()
        assert np.allclose(probs, probs[0], atol=1e-12)


def test_mutated_candidate_probability_is_rejected(sample):
    future = sample["future"].copy()
    selection = sample["selection"].copy()
    future.loc[future.index[0], "selected_low"] += .01
    with pytest.raises(DashboardDataError, match="Некоректно обрані"):
        verify_model_choices(future, selection)


def test_calibration_periods_precede_scored_target_periods(sample):
    verify_frozen_windows(sample["selection"], sample["thresholds"], sample["manifest"])
    broken = dict(sample["manifest"])
    broken["selection_threshold_reference_180days_through"] = "2026-09-11"
    with pytest.raises(DashboardDataError):
        verify_frozen_windows(sample["selection"], sample["thresholds"], broken)


def test_validation_labels_cannot_change_selection_thresholds(sample):
    dates = pd.date_range("2025-01-01", "2026-10-09", freq="D")
    s = pd.Series(np.tile([1., 3., 5., 8., 12., 4., 9.], len(dates)//7+1)[:len(dates)],
                  index=dates)
    before = selection_thresholds(s)
    altered = s.copy()
    altered.loc["2026-08-15":"2026-10-09"] = 24.0
    after = selection_thresholds(altered)
    assert before == after


def test_holdout_cannot_modify_final_region_thresholds(sample):
    daily = sample["daily"].copy()
    daily["date"] = daily["date"].dt.date  # Match the Stage 2 input contract.
    before, _, _ = build_training_regimes(daily, CANDIDATE_REGIONS, TRAIN_START, TRAIN_END)
    edited = daily.copy()
    mask = edited["date"] >= date(2026, 9, 12)
    edited.loc[mask, "recorded_hours"] = 0.0
    after, _, _ = build_training_regimes(edited, CANDIDATE_REGIONS, TRAIN_START, TRAIN_END)
    pd.testing.assert_frame_equal(before, after)


def test_region_series_stops_at_forecast_origin(sample):
    region = CANDIDATE_REGIONS[0]
    daily = sample["daily"].copy()
    base = get_region_series(daily.assign(date=daily["date"].dt.date), region, date(2026, 9, 11))
    modified = daily.assign(date=daily["date"].dt.date).copy()
    mask = (modified["region"] == region) & (modified["date"] > date(2026, 9, 11))
    modified.loc[mask, "recorded_hours"] = 23.
    compared = get_region_series(modified, region, date(2026, 9, 11))
    pd.testing.assert_series_equal(base, compared)
