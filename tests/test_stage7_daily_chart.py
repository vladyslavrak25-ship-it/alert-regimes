"""Stage 7: calendar-only x labels, width, data fidelity, no silent imputation."""
from datetime import date
import json

import pandas as pd
import pytest

from history_daily_chart import daily_columns_spec


def series(year=2026, month=1, days=(1, 2, 15, 31), hours=(1.2, 4.6, 0.0, 10.1)):
    return pd.DataFrame({
        'date': pd.to_datetime([date(year,month,d) for d in days]),
        'recorded_hours': list(hours),
    })


def test_every_calendar_day_is_shown_as_a_category_with_no_clock_ticks():
    spec = daily_columns_spec(series(), 2026, 1)
    x = spec['encoding']['x']
    assert x['type'] == 'ordinal'
    assert x['field'] == 'day_label'
    assert x['scale']['domain'] == [f'{d:02d}' for d in range(1,32)]
    assert x['axis']['values'] == [f'{d:02d}' for d in range(1,32)]
    assert x['axis']['labelAngle'] == 0
    assert x['axis']['labelOverlap'] is False
    assert '06 AM' not in json.dumps(spec)
    assert not spec.get('selection')
    # Only calendar labels, not temporal or datetime fields.
    assert all(e['type'] != 'temporal' for e in (spec['encoding']['x'],*spec['encoding']['tooltip']))


def test_bars_are_wider_but_separate_and_y_is_still_hours():
    spec = daily_columns_spec(series(), 2026, 1)
    assert spec['mark']['type'] == 'bar'
    assert 18 <= spec['mark']['size'] < spec['width']['step']
    assert spec['encoding']['y']['field'] == 'recorded_hours'
    assert spec['encoding']['y']['type'] == 'quantitative'
    assert spec['encoding']['y']['scale']['zero'] is True
    assert 'год' in spec['encoding']['y']['axis']['title']
    assert [v['field'] for v in spec['encoding']['tooltip']] == ['date_label','recorded_hours']


def test_no_missing_days_imputed_or_zeros_turned_into_missing():
    spec = daily_columns_spec(series(), 2026, 1)
    vals = spec['data']['values']
    assert len(vals) == 4
    assert {r['day_label'] for r in vals} == {'01','02','15','31'}
    assert next(r['recorded_hours'] for r in vals if r['day_label']=='15') == 0
    assert '03' in spec['encoding']['x']['scale']['domain']
    assert '03' not in {r['day_label'] for r in vals}
    assert vals[0]['date_label'] == '01.01.2026'
    json.dumps(spec, allow_nan=False)  # Valid serializable Vega-Lite data.


def test_leap_year_february_and_30_day_month():
    leap = daily_columns_spec(series(2024,2,(29,), (3.5,)), 2024,2)
    assert len(leap['encoding']['x']['scale']['domain']) == 29
    april = daily_columns_spec(series(2026,4,(30,), (1.2,)), 2026,4)
    assert len(april['encoding']['x']['scale']['domain']) == 30


def test_invalid_daily_values_and_duplicate_dates_are_rejected():
    bad = series()
    bad.loc[0,'recorded_hours'] = -1
    with pytest.raises(ValueError,match='0–24'):
        daily_columns_spec(bad,2026,1)
    bad = series()
    bad.loc[0,'recorded_hours'] = float('nan')
    with pytest.raises(ValueError,match='години'):
        daily_columns_spec(bad,2026,1)
    bad = series()
    bad.loc[1,'date'] = bad.loc[0,'date']
    with pytest.raises(ValueError,match='дублюється'):
        daily_columns_spec(bad,2026,1)


def test_wrong_month_and_25_hours_rejected():
    with pytest.raises(ValueError,match='поза вибраним місяцем'):
        daily_columns_spec(series(),2026,2)
    with pytest.raises(ValueError,match='0–24'):
        daily_columns_spec(series(hours=(1,2,25,4)),2026,1)
