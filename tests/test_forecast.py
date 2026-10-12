"""Stage 3 tests: designed to catch look-ahead and forecast-date mistakes."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast import (
    CLASSES, ForecastError, baseline_probabilities, features_from_origin,
    fit_probabilistic_model, multiclass_brier, predict_for_dates,
    get_region_series,
)


def example_series(start="2025-01-01", n=550):
    """Artificial complete history with three regimes; never real alerts."""
    times = pd.date_range(start, periods=n, freq="D")
    values = np.array([2.0, 4.5, 7.5, 5.0, 8.0, 1.0, 3.5] * ((n+6)//7))[:n]
    return pd.Series(values, index=times)


def test_baseline_sums_to_one_and_has_three_probabilities():
    result = baseline_probabilities(example_series().iloc[:50], 3, 6)
    assert result.shape == (3,)
    assert np.all(result > 0)
    assert result.sum() == pytest.approx(1.0)


def test_features_reject_wrong_target_date():
    history = example_series().iloc[:60]
    next_day = history.index[-1].date() + timedelta(days=1)
    with pytest.raises(ForecastError, match="не відповідає"):
        features_from_origin(history, next_day, 7, 3, 6)


def test_features_never_change_if_future_changes():
    series = example_series()
    as_of = series.index[399].date()
    hist = series.loc[:pd.Timestamp(as_of)]
    target = as_of + timedelta(days=6)
    before = features_from_origin(hist, target, 6, 3, 6)
    changed = series.copy()
    changed.iloc[400:] = 24.0
    after = features_from_origin(changed.loc[:pd.Timestamp(as_of)], target, 6, 3, 6)
    np.testing.assert_array_equal(before, after)


def test_probabilities_for_seven_future_days_use_same_origin():
    series = example_series()
    as_of = series.index[499].date()
    model = fit_probabilistic_model(series, 3, 6, as_of)
    targets = [as_of + timedelta(days=i) for i in range(3, 10)]
    predicted = predict_for_dates(model, series, as_of, targets, 3, 6)
    assert len(predicted) == 7
    assert predicted["horizon"].tolist() == list(range(3, 10))
    assert predicted["origin"].nunique() == 1
    assert np.allclose(predicted[["low","medium","high"]].sum(axis=1), 1)
    assert np.allclose(predicted[["baseline_low","baseline_medium","baseline_high"]].sum(axis=1), 1)


def test_model_does_not_fit_on_values_after_cutoff():
    series = example_series()
    as_of = series.index[500].date()
    base_model = fit_probabilistic_model(series, 3, 6, as_of)
    future_changed = series.copy()
    future_changed.iloc[501:] = 24.0
    changed_model = fit_probabilistic_model(future_changed, 3, 6, as_of)
    np.testing.assert_allclose(base_model[-1].coef_, changed_model[-1].coef_, atol=1e-12)
    np.testing.assert_allclose(base_model[-1].intercept_, changed_model[-1].intercept_, atol=1e-12)


def test_brier_score_prefers_correct_confident_forecast():
    assert multiclass_brier(np.array([1.0,0.0,0.0]), "low") == pytest.approx(0)
    assert multiclass_brier(np.array([0.0,1.0,0.0]), "low") == pytest.approx(2)
    assert multiclass_brier(np.array([1/3]*3), "low") == pytest.approx(2/3)


def test_missing_calendar_row_is_not_silently_turned_to_zero():
    synthetic = pd.DataFrame({
        "date": list(pd.date_range("2026-01-01", periods=31).date),
        "region": "Sumska oblast",
        "recorded_hours": 2.0,
    })
    synthetic = synthetic.loc[synthetic["date"] != date(2026,1,12)]
    with pytest.raises(ForecastError, match="відсутній"):
        get_region_series(synthetic, "Sumska oblast", date(2026,1,31))


def test_asof_requires_last_day_and_cannot_use_later_date():
    synthetic = pd.DataFrame({
        "date":list(pd.date_range("2026-01-01", periods=35).date),
        "region":"Sumska oblast", "recorded_hours":2.0,
    })
    with pytest.raises(ForecastError, match="немає даних"):
        get_region_series(synthetic, "Sumska oblast", date(2026,3,1))
