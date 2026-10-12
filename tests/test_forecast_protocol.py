"""New tests for Stage 3 correction: model selection MUST precede holdout."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import forecast
from forecast import (
    choose_model, evaluate_experiment, selection_thresholds,
    selected_probabilities, ForecastError,
)


def synthetic_series():
    # Synthetic chronological sample, not actual Ukraine alert data.
    ix = pd.date_range("2024-01-01", "2026-10-09", freq="D")
    values = np.resize([1.0, 3.0, 4.0, 5.0, 8.0, 6.0, 2.0, 7.0], len(ix))
    return pd.Series(values, index=ix)


def test_selection_does_not_depend_on_holdout_values():
    s = synthetic_series()
    a = selection_thresholds(s)
    altered = s.copy()
    altered.loc["2026-09-12":] = 23
    assert selection_thresholds(altered) == a


def test_selection_tie_prefers_baseline():
    same = pd.DataFrame({"model_brier": [0.5, 0.6], "baseline_brier": [0.5, 0.6]})
    assert choose_model(same) == "baseline"
    same["model_brier"] = [0.3, 0.4]
    assert choose_model(same) == "logistic"


def test_selected_probabilities_are_exactly_chosen_model():
    f = pd.DataFrame({"model_low": [0.1], "model_medium": [0.2], "model_high": [0.7],
                      "baseline_low": [0.5], "baseline_medium": [0.4],
                      "baseline_high": [0.1], "model_brier": [0.5],
                      "baseline_brier": [0.9], "actual_regime": ["high"]})
    output = selected_probabilities(f, "baseline")
    assert output.loc[0, "selected_high"] == pytest.approx(0.1)
    assert output.loc[0, "selected_brier"] == pytest.approx(0.9)
    assert selected_probabilities(f, "logistic").loc[0, "selected_high"] == pytest.approx(0.7)


def test_experiment_has_unique_dates_and_expected_horizons(monkeypatch):
    s = synthetic_series()
    class FakeModel:
        classes_ = np.array(["low", "medium", "high"])
        def predict_proba(self, X):
            return np.array([[0.25, 0.50, 0.25]])
    fit_calls = []
    def fake_fit(series, lo, hi, origin):
        fit_calls.append(origin)
        return FakeModel()
    monkeypatch.setattr(forecast, "fit_probabilistic_model", fake_fit)
    origins = forecast.TEST_ORIGINS
    a = evaluate_experiment(s, 3, 6, origins,
                            date(2026, 9, 12), date(2026, 10, 9), 1, "A")
    b = evaluate_experiment(s, 3, 6, origins,
                            date(2026, 9, 12), date(2026, 10, 9), 3, "B")
    assert len(a) == 28 and a.date.nunique() == 28
    assert len(b) == 26 and b.date.nunique() == 26
    assert set(a.horizon) == set(range(1, 8))
    assert set(b.horizon) == set(range(3, 10))
    assert fit_calls == list(origins) + list(origins)
    assert (pd.to_datetime(a.origin) < pd.to_datetime(a.date)).all()
    assert (pd.to_datetime(b.origin) < pd.to_datetime(b.date)).all()


def test_evaluate_never_uses_target_future_as_model_features(monkeypatch):
    s = synthetic_series()
    class FakeModel:
        classes_ = np.array(["low", "medium", "high"])
        def predict_proba(self, X):
            return np.array([[0.2, 0.3, 0.5]])
    monkeypatch.setattr(forecast, "fit_probabilistic_model", lambda *args: FakeModel())
    origin = (date(2026, 9, 11),)
    before = evaluate_experiment(s, 3, 6, origin, date(2026, 9, 12),
                                  date(2026, 9, 18), 1, "A")
    changed = s.copy()
    changed.loc["2026-09-12":] = 22.0
    after = evaluate_experiment(changed, 3, 6, origin, date(2026, 9, 12),
                                 date(2026, 9, 18), 1, "A")
    for col in ("model_low", "model_medium", "model_high",
                "baseline_low", "baseline_medium", "baseline_high"):
        np.testing.assert_array_equal(before[col].to_numpy(), after[col].to_numpy())
    assert not (before["actual_regime"] == after["actual_regime"]).all()


def test_repeat_dates_raise_error(monkeypatch):
    s = synthetic_series()
    class FakeModel:
        classes_ = np.array(["low", "medium", "high"])
        def predict_proba(self, X):
            return np.array([[0.2, 0.3, 0.5]])
    monkeypatch.setattr(forecast, "fit_probabilistic_model", lambda *args: FakeModel())
    with pytest.raises(ForecastError, match="Повторна"):
        evaluate_experiment(s, 3, 6, (date(2026, 9, 11), date(2026, 9, 12)),
                            date(2026, 9, 12), date(2026, 9, 20), 1, "overlap")
