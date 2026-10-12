"""Monthly, comparable history for seven regions; no model training or label changes."""
from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd

from dashboard_data import DashboardDataError, REGION_UK

MONTH_UK = (
    'Січ', 'Лют', 'Бер', 'Кві', 'Тра', 'Чер',
    'Лип', 'Сер', 'Вер', 'Жов', 'Лис', 'Гру',
)


def month_summary(full_history: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Mean HOURS per *available calendar day*, not sum or mean over alert events.

    An incomplete month is retained but clearly labeled via `is_full_month`.
    A month with no source interval still counts as a calendar day in the
    prepared daily data. Missing region-months remain missing (not zero).
    """
    if not regions or len(set(regions)) != len(regions):
        raise DashboardDataError('Для помісячної історії потрібні неповторні області.')
    required = {'date', 'region', 'recorded_hours'}
    if not required.issubset(full_history.columns):
        raise DashboardDataError('У добовій таблиці не вистачає полів.')
    part = full_history.loc[full_history['region'].isin(regions), list(required)].copy()
    if part.empty or part.duplicated(['region', 'date']).any():
        raise DashboardDataError('В історії немає рядків або є повторені дні.')
    part['date'] = pd.to_datetime(part['date'])
    values = pd.to_numeric(part['recorded_hours'], errors='coerce')
    if not np.isfinite(values).all() or (values < 0).any() or (values > 24.00001).any():
        raise DashboardDataError('Неправильна кількість годин у добовій таблиці.')
    part['recorded_hours'] = values
    part['year'] = part['date'].dt.year
    part['month'] = part['date'].dt.month
    agg = (part.groupby(['region','year','month'], as_index=False)
           .agg(mean_hours=('recorded_hours','mean'),
                total_hours=('recorded_hours','sum'),
                observed_days=('date','size'),
                first_date=('date','min'),
                last_date=('date','max')))
    agg['calendar_days'] = [calendar.monthrange(int(y), int(m))[1]
                            for y,m in zip(agg['year'],agg['month'])]
    agg['is_full_month'] = (
        (agg['observed_days'] == agg['calendar_days']) &
        (agg['first_date'].dt.day == 1) &
        (agg['last_date'].dt.day == agg['calendar_days'])
    )
    # Catch gaps, do not imply incomplete middle months are complete.
    if ((agg['total_hours'] / agg['observed_days'] - agg['mean_hours']).abs() > 1e-10).any():
        raise DashboardDataError('Помилка помісячної агрегації.')
    agg['Область'] = agg['region'].map(REGION_UK).fillna(agg['region'])
    return agg.sort_values(['year','month','region']).reset_index(drop=True)


def year_heatmap(summary: pd.DataFrame, regions: list[str], year: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Return 7×12 month means and observed-day counts in the same order.

    Missing region-month -> NaN for means and counts, NOT a zero-hour month.
    """
    year_data = summary.loc[summary['year'] == year]
    means = year_data.pivot(index='region', columns='month', values='mean_hours')
    days = year_data.pivot(index='region', columns='month', values='observed_days')
    means = means.reindex(index=regions, columns=range(1, 13))
    days = days.reindex(index=regions, columns=range(1, 13))
    if means.isna().all().all():
        raise DashboardDataError(f'Немає помісячних даних за {year} рік.')
    means.index = [REGION_UK.get(r,r) for r in means.index]
    days.index = means.index
    means.columns = MONTH_UK
    days.columns = MONTH_UK
    return means, days


def month_details(full_history: pd.DataFrame, region: str, year: int, month: int) -> pd.DataFrame:
    dates = pd.to_datetime(full_history['date'])
    one = full_history.loc[
        (full_history['region'] == region) & (dates.dt.year == year) & (dates.dt.month == month),
        ['date', 'recorded_hours', 'has_source_interval', 'naive_added_hours'],
    ].copy()
    if one.empty:
        raise DashboardDataError('Для вибраної області та місяця немає добових рядків.')
    one['date'] = pd.to_datetime(one['date'])
    if len(one) > 31 or one['date'].duplicated().any():
        raise DashboardDataError('Некоректна деталізація місяця.')
    return one.sort_values('date').reset_index(drop=True)
