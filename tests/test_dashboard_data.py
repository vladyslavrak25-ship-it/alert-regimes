"""Stage 4 tests of read-only UI data contracts; Streamlit not required."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard_data import (  # noqa: E402
    DashboardDataError, check_probabilities, classify_daily,
    load_model_selection,
)


def test_selection_loads_headerless_csv(tmp_path):
    path = tmp_path / "model_selection.csv"
    path.write_text("Sumska oblast,baseline,2026-09-11,0.8,0.6,26\n", encoding="utf-8")
    selection = load_model_selection(path)
    assert selection.iloc[0]["region"] == "Sumska oblast"
    assert selection.iloc[0]["selected_model"] == "baseline"


def test_selection_loads_header_csv(tmp_path):
    path = tmp_path / "model_selection.csv"
    path.write_text(
        "region,selected_model,selection_end,selection_gap_brier_logistic,selection_gap_brier_baseline,selection_forecasts\n"
        "Sumska oblast,logistic,2026-09-11,0.4,0.5,26\n", encoding="utf-8"
    )
    assert load_model_selection(path).iloc[0]["selected_model"] == "logistic"


def test_classify_regimes_with_exact_boundaries():
    r = classify_daily(pd.Series([1.0, 3.0, 3.01, 5.0, 5.01]), 3.0, 5.0)
    assert list(r) == ["low", "low", "medium", "medium", "high"]


def test_valid_probabilities():
    p = pd.DataFrame({"selected_low": [0.2], "selected_medium": [0.5], "selected_high": [0.3]})
    check_probabilities(p)


def test_invalid_probabilities_fail_loudly():
    p = pd.DataFrame({"selected_low": [0.2], "selected_medium": [0.5], "selected_high": [0.5]})
    with pytest.raises(DashboardDataError, match="Сума ймовірностей"):
        check_probabilities(p)


def test_duplicate_regions_in_selection_fail(tmp_path):
    path = tmp_path / "model_selection.csv"
    path.write_text(
        "Sumska oblast,baseline,2026-09-11,0.8,0.6,26\n"
        "Sumska oblast,baseline,2026-09-11,0.8,0.6,26\n", encoding="utf-8")
    with pytest.raises(DashboardDataError, match="повтори"):
        load_model_selection(path)
