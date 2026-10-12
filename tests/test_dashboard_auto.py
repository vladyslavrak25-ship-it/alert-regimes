from pathlib import Path
from datetime import timedelta
import json

import pandas as pd
import pytest

from dashboard_auto import load_latest_week, AutoForecastError


def _sample(tmp_path: Path):
    start = pd.Timestamp('2026-10-19')
    names = {'Chernihivska oblast': 'baseline', 'Poltavska oblast': 'logistic',
             'Sumska oblast': 'baseline', 'Zaporizka oblast': 'baseline',
             'Kharkivska oblast': 'baseline', 'Dnipropetrovska oblast': 'baseline',
             'Mykolaivska oblast': 'logistic'}
    rows = []
    for r, m in names.items():
        for i in range(7):
            rows.append(dict(region=r, date=(start + pd.Timedelta(days=i)).date(),
                             origin='2026-10-17', horizon=i+2, selected_model=m,
                             selected_low=.2, selected_medium=.3, selected_high=.5))
    f = pd.DataFrame(rows)
    f.to_csv(tmp_path/'auto_week_latest.csv', index=False)
    meta = dict(forecast_first_day='2026-10-19', forecast_last_day='2026-10-25', last_training_day='2026-10-17')
    (tmp_path/'auto_week_latest_manifest.json').write_text(json.dumps(meta))
    return f,names


def test_no_auto_is_valid(tmp_path):
    assert load_latest_week(tmp_path, {}) is None


def test_reads_seven_day_auto_with_provenance(tmp_path):
    _, names = _sample(tmp_path)
    loaded, meta = load_latest_week(tmp_path, names)
    assert len(loaded) == 49
    assert meta['last_training_day'] == '2026-10-17'


def test_refuses_leaked_origin(tmp_path):
    _, names = _sample(tmp_path)
    path = tmp_path/'auto_week_latest_manifest.json'
    meta = json.loads(path.read_text())
    meta['last_training_day'] = '2026-10-19'
    path.write_text(json.dumps(meta))
    with pytest.raises(AutoForecastError):
        load_latest_week(tmp_path, names)


def test_refuses_wrong_model(tmp_path):
    frame, names = _sample(tmp_path)
    frame.loc[0, 'selected_model'] = 'logistic'
    frame.to_csv(tmp_path/'auto_week_latest.csv', index=False)
    with pytest.raises(AutoForecastError):
        load_latest_week(tmp_path, names)


def test_refuses_bad_probabilities(tmp_path):
    frame, names = _sample(tmp_path)
    frame.loc[0, 'selected_high'] = 2
    frame.to_csv(tmp_path/'auto_week_latest.csv', index=False)
    with pytest.raises(AutoForecastError):
        load_latest_week(tmp_path, names)
