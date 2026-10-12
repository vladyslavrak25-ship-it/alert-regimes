"""Stage 6: monthly heatmap fidelity and honest earlier-window comparisons."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dashboard_data import DashboardDataError, load_outputs
from history_heatmap import MONTH_UK, month_summary, year_heatmap, month_details
from model_robustness import (
    FIRST_ORIGINS, TARGET_REGIONS, historical_thresholds, fold_geometry,
    summarize_folds,
)

OUT = Path(__file__).resolve().parents[1] / 'outputs'


def test_partial_month_and_missing_month_are_not_zero():
    frame = pd.DataFrame({
        'region': ['Chernihivska oblast']*2 + ['Sumska oblast']*2,
        'date': pd.to_datetime(['2022-02-25','2022-02-26', '2022-03-01', '2022-03-02']),
        'recorded_hours': [2., 6., 0., 2.],
    })
    summary = month_summary(frame, ['Chernihivska oblast','Sumska oblast'])
    assert not summary['is_full_month'].any()
    means, days = year_heatmap(summary, ['Chernihivska oblast','Sumska oblast'], 2022)
    assert means.loc['Чернігівська область', 'Лют'] == 4.
    assert days.loc['Чернігівська область', 'Лют'] == 2
    assert np.isnan(means.loc['Чернігівська область', 'Бер'])
    assert means.loc['Сумська область', 'Бер'] == 1.
    assert days.loc['Сумська область', 'Бер'] == 2
    assert list(means.columns) == list(MONTH_UK)


def test_complete_leap_february_is_marked_complete():
    dates = pd.date_range('2024-02-01','2024-02-29',freq='D')
    frame = pd.DataFrame({'region':['Chernihivska oblast']*29,
                          'date':dates, 'recorded_hours':[3.]*29})
    agg = month_summary(frame, ['Chernihivska oblast'])
    assert agg.iloc[0]['calendar_days'] == 29
    assert bool(agg.iloc[0]['is_full_month'])
    assert agg.iloc[0]['observed_days'] == 29


def test_duplicate_month_input_fails_closed():
    frame = pd.DataFrame({'region':['Sumska oblast']*2,
                         'date':pd.to_datetime(['2026-01-01']*2),
                         'recorded_hours':[2., 5.]})
    with pytest.raises(DashboardDataError, match='повторені'):
        month_summary(frame, ['Sumska oblast'])


def test_monthly_aggregate_matches_original_daily(sample=None):
    data = load_outputs(OUT)
    regions = list(data['selection']['region'])
    cutoff = pd.Timestamp(data['manifest']['date_of_last_complete_observation'])
    raw = data['daily'].loc[data['daily']['region'].isin(regions) &
                            (data['daily']['date'] <= cutoff)]
    agg = month_summary(raw, regions)
    assert set(agg['region']) == set(regions)
    assert len(agg) > 300
    assert agg['observed_days'].sum() == len(raw)
    assert abs(agg['total_hours'].sum() - raw['recorded_hours'].sum()) < 0.00001
    assert agg['first_date'].min().year == 2022
    # October 2026 is partial: only through Oct 9.
    october = agg.loc[(agg['year']==2026)&(agg['month']==10)]
    assert len(october) == 7
    assert (october['observed_days'] == 9).all()
    assert not october['is_full_month'].any()


def test_31_day_drill_down_matches_month_cell():
    data = load_outputs(OUT)
    region = TARGET_REGIONS[0]
    raw = data['daily'].loc[data['daily']['region'] == region]
    detail = month_details(raw,region,2026,5)
    assert len(detail) == 31
    agg = month_summary(raw,[region])
    May = agg.loc[(agg['year']==2026)&(agg['month']==5)].iloc[0]
    assert detail['recorded_hours'].mean() == pytest.approx(May['mean_hours'])


def test_fold_geometries_contiguous_nonoverlapping_prior_to_selection():
    target_dates = []
    for first in FIRST_ORIGINS:
        origins, start, end = fold_geometry(first)
        assert len(origins) == 4
        assert (start-first).days == 3
        assert (end-start).days == 27
        assert (end < date(2026,8,15))
        target_dates.extend([start+timedelta(days=i) for i in range(28)])
    assert len(target_dates) == len(set(target_dates)) == 112
    assert target_dates[-1] == date(2026,8,9)


def test_thresholds_are_immune_to_target_and_later_days():
    dates = pd.date_range('2024-01-01','2026-09-30')
    values = np.tile(np.array([1.,2.,3.,4.,5.,6.,7.]), (len(dates)//7)+1)[:len(dates)]
    series = pd.Series(values,index=dates)
    first = FIRST_ORIGINS[0]
    before = historical_thresholds(series, first)
    changed = series.copy()
    changed.loc[pd.Timestamp(first)+pd.Timedelta(days=1):] = 24.
    assert historical_thresholds(changed, first) == before


def test_summary_weights_folds_by_observed_target_counts():
    frame = pd.DataFrame({
        'region':['Chernihivska oblast']*2,
        'target_days':[28,14],
        'baseline_brier':[.4,.8],
        'logistic_brier':[.5,.6],
    })
    table = summarize_folds(frame)
    expected_base = (28*.4+14*.8)/42
    expected_log = (28*.5+14*.6)/42
    assert table.iloc[0]['target_days'] == 42
    assert table.iloc[0]['baseline_brier'] == pytest.approx(expected_base)
    assert table.iloc[0]['logistic_brier'] == pytest.approx(expected_log)
    assert table.iloc[0]['folds_logistic_better'] == 1
