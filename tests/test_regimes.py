"""Independent, small cases for our three activity regimes."""

from datetime import date, timedelta

import pandas as pd
import pytest

from regimes import (
    RegimeDataError,
    build_training_regimes,
    classify_hours,
    read_daily_table,
    training_only,
)


def example_daily(start=date(2026, 1, 1), n=180, region="Kharkivska oblast"):
    """Small artificial pattern: three different activity levels, no real events."""
    rows = []
    for i in range(n):
        hours = [2.0, 4.0, 6.0, 8.0][i % 4]
        rows.append({
            "region": region,
            "date": start + timedelta(days=i),
            "recorded_hours": hours,
            "naive_added_hours": 0.0,
            "has_source_interval": True,
        })
    return pd.DataFrame(rows)


def test_limits_include_the_boundary_in_lower_class():
    assert classify_hours(2.0, 2.0, 6.0) == "low"
    assert classify_hours(2.01, 2.0, 6.0) == "medium"
    assert classify_hours(6.0, 2.0, 6.0) == "medium"
    assert classify_hours(6.01, 2.0, 6.0) == "high"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(RegimeDataError):
        classify_hours(3, 5, 5)


def test_train_only_does_not_include_future_days():
    df = example_daily(n=200)
    start, end = date(2026, 1, 1), date(2026, 6, 29)  # 180 days
    train = training_only(df, regions=("Kharkivska oblast",), start=start, end=end)
    assert len(train) == 180
    assert train["date"].max() == end


def test_mutating_future_does_not_change_train_thresholds():
    df = example_daily(n=208)
    start, end = date(2026, 1, 1), date(2026, 6, 29)
    before, _, _ = build_training_regimes(df, ("Kharkivska oblast",), start, end)
    changed = df.copy()
    changed.loc[changed["date"] > end, "recorded_hours"] = 24.0
    after, _, _ = build_training_regimes(changed, ("Kharkivska oblast",), start, end)
    pd.testing.assert_frame_equal(before, after)


def test_missing_calendar_row_is_not_replaced_with_zero():
    df = example_daily()
    df = df.loc[df["date"] != date(2026, 1, 12)]
    with pytest.raises(RegimeDataError, match="відсутні"):
        training_only(df, ("Kharkivska oblast",), date(2026, 1, 1), date(2026, 6, 29))


def test_three_classes_present_in_good_training_sample():
    df = example_daily()
    thresholds, labels, report = build_training_regimes(
        df, ("Kharkivska oblast",), date(2026, 1, 1), date(2026, 6, 29)
    )
    assert len(thresholds) == 1
    assert set(labels["regime"]) == {"low", "medium", "high"}
    assert bool(report.iloc[0]["suitable"])


def test_quality_check_skips_region_with_too_few_active_days():
    good = example_daily(region="Kharkivska oblast")
    bad = example_daily(region="Mykolaivska oblast")
    bad.loc[:30, "recorded_hours"] = 0.0
    bad.loc[:30, "has_source_interval"] = False
    df = pd.concat([good, bad], ignore_index=True)
    th, labels, readiness = build_training_regimes(
        df, ("Kharkivska oblast", "Mykolaivska oblast"),
        date(2026, 1, 1), date(2026, 6, 29)
    )
    assert th["region"].tolist() == ["Kharkivska oblast"]
    assert set(labels["region"]) == {"Kharkivska oblast"}
    assert not bool(readiness.loc[readiness.region == "Mykolaivska oblast", "suitable"].iloc[0])


def test_daily_csv_rejects_duplicate_region_day(tmp_path):
    df = example_daily(n=2)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    f = tmp_path / "daily.csv"
    df.to_csv(f, index=False)
    with pytest.raises(RegimeDataError, match="Дублікат"):
        read_daily_table(f)
