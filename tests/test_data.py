"""Synthetic cases with answers we can verify manually."""

from datetime import date

import pandas as pd
import pytest

from data import DataQualityError, clean_records, daily_hours


def calculated(intervals, through):
    """Build small CSV-like inputs in the same format as the real dataset."""
    raw = pd.DataFrame(intervals, columns=["region", "started_at", "finished_at", "naive"])
    clean, _ = clean_records(raw)
    return daily_hours(clean, through).set_index(["region", "date"])


def get_hours(out, day, field="recorded_hours", region="Kyiv City"):
    return float(out.loc[(region, day), field])


def test_cross_midnight_local_kyiv():
    # 23:30 in Kyiv on Oct 8 is 20:30 UTC, then 01:00 on Oct 9 is 22:00 UTC.
    out = calculated([
        ("Kyiv City", "2026-10-08T20:30:00Z", "2026-10-08T22:00:00Z", False)
    ], date(2026, 10, 9))
    assert get_hours(out, "2026-10-08") == pytest.approx(0.5)
    assert get_hours(out, "2026-10-09") == pytest.approx(1.0)


def test_overlapping_intervals_are_not_double_counted():
    out = calculated([
        ("Kyiv City", "2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", False),
        ("Kyiv City", "2026-08-01T10:30:00Z", "2026-08-01T11:30:00Z", True),
    ], date(2026, 8, 1))
    assert get_hours(out, "2026-08-01") == pytest.approx(1.5)
    assert get_hours(out, "2026-08-01", "without_naive_hours") == pytest.approx(1.0)
    assert get_hours(out, "2026-08-01", "naive_added_hours") == pytest.approx(0.5)


def test_exact_duplicates_removed():
    row = ("Kyiv City", "2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", False)
    clean, audit = clean_records(pd.DataFrame([row, row], columns=[
        "region", "started_at", "finished_at", "naive"
    ]))
    assert len(clean) == 1
    assert audit["exact_duplicate_rows_removed"] == 1


def test_invalid_zero_length_is_reported_and_removed():
    rows = [
        ("Kyiv City", "2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", False),
    ] * 99 + [
        ("Kyiv City", "2026-08-01T12:00:00Z", "2026-08-01T12:00:00Z", False)
    ]
    _, audit = clean_records(pd.DataFrame(rows, columns=[
        "region", "started_at", "finished_at", "naive"
    ]))
    assert audit["nonpositive_interval_rows"] == 1
    assert audit["invalid_rows_removed"] == 1


def test_missing_column_is_clear_error():
    with pytest.raises(DataQualityError, match="Відсутні стовпці"):
        clean_records(pd.DataFrame({"region": ["Kyiv City"]}))


def test_unexpected_region_is_not_silently_accepted():
    with pytest.raises(DataQualityError, match="Невідомі регіони"):
        calculated([
            ("Unknown Area", "2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", False)
        ], date(2026, 8, 1))


def test_empty_day_is_not_claimed_to_be_verified_zero():
    out = calculated([
        ("Kyiv City", "2026-08-01T10:00:00Z", "2026-08-01T11:00:00Z", False)
    ], date(2026, 8, 3))
    assert get_hours(out, "2026-08-02") == 0
    assert not bool(out.loc[("Kyiv City", "2026-08-02"), "has_source_interval"])


def test_dst_fall_back_day_can_have_25_hours():
    # On Oct 25, 2026 Kyiv switches from UTC+3 to UTC+2 (per tz database).
    out = calculated([
        ("Kyiv City", "2026-10-24T21:00:00Z", "2026-10-25T22:00:00Z", False)
    ], date(2026, 10, 25))
    assert get_hours(out, "2026-10-25") == pytest.approx(25.0)


def test_no_future_day_in_output():
    out = calculated([
        ("Kyiv City", "2026-08-01T10:00:00Z", "2026-08-03T10:00:00Z", False)
    ], date(2026, 8, 1))
    assert set(day for _, day in out.index) == {"2026-08-01"}
    assert get_hours(out, "2026-08-01") == pytest.approx(11.0)  # 13:00–00:00 Kyiv
