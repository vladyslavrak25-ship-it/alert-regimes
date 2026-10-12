"""Pure Vega-Lite daily chart builder: calendar days, never a datetime x-axis.

No Streamlit or Altair import is needed here. The app renders the resulting
Vega-Lite specification with st.vega_lite_chart(). This module does not change
or calculate any forecasts or regime categories.
"""
from __future__ import annotations

import calendar
import math

import pandas as pd


def daily_columns_spec(details: pd.DataFrame, year: int, month: int) -> dict:
    """Build a legible 1-month bar plot with 01..28/29/30/31 categorical ticks.

    Days with no daily row are left blank, NOT imputed as zero. Observed zero
    hours remain exactly zero in the chart data. ``recorded_hours`` is in hours.
    """
    expected = {"date", "recorded_hours"}
    if not expected.issubset(details.columns):
        raise ValueError("Деталізація має містити дату й записані години.")
    if details.empty:
        raise ValueError("Не можна побудувати графік без добових рядків.")

    year = int(year)
    month = int(month)
    n_days = calendar.monthrange(year, month)[1]
    dates = pd.to_datetime(details["date"], errors="raise")
    if dates.isna().any():
        raise ValueError("У деталізації є порожні дати.")
    if dates.dt.year.ne(year).any() or dates.dt.month.ne(month).any():
        raise ValueError("У деталізації є дні поза вибраним місяцем.")
    if dates.dt.day.duplicated().any():
        raise ValueError("Один календарний день дублюється.")

    hours = pd.to_numeric(details["recorded_hours"], errors="coerce")
    if hours.isna().any() or not hours.map(math.isfinite).all():
        raise ValueError("Некоректні добові години.")
    if (hours < 0).any() or (hours > 24.00001).any():
        raise ValueError("Добова тривалість поза межами 0–24 год.")

    values = [
        {
            "day_label": f"{int(d.day):02d}",
            "date_label": d.strftime("%d.%m.%Y"),
            "recorded_hours": float(h),
        }
        for d, h in sorted(zip(dates, hours), key=lambda item: item[0])
    ]
    day_labels = [f"{day:02d}" for day in range(1, n_days + 1)]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Добові години тривог; по X тільки календарні дні, без часової шкали.",
        # 32 px per day: 22 px columns + room between bars, including 31-day months.
        "width": {"step": 32},
        "height": 325,
        "data": {"values": values},
        "mark": {
            "type": "bar",
            "size": 22,
            "color": "#3478BD",
            "cornerRadiusTopLeft": 3,
            "cornerRadiusTopRight": 3,
        },
        "encoding": {
            "x": {
                "field": "day_label",
                "type": "ordinal",
                "sort": day_labels,
                "scale": {"domain": day_labels},
                "axis": {
                    "title": "День місяця",
                    "values": day_labels,
                    "labelAngle": 0,
                    "labelOverlap": False,
                    "labelFontSize": 11,
                    "labelPadding": 9,
                    "tickSize": 0,
                },
            },
            "y": {
                "field": "recorded_hours",
                "type": "quantitative",
                "scale": {"zero": True, "nice": True},
                "axis": {"title": "Тривалість, год", "tickCount": 6},
            },
            "tooltip": [
                {"field": "date_label", "type": "nominal", "title": "Дата"},
                {"field": "recorded_hours", "type": "quantitative", "title": "Годин за добу", "format": ".2f"},
            ],
        },
        "config": {
            "view": {"stroke": None},
            "axis": {"gridColor": "#E5EAF0"},
        },
    }
