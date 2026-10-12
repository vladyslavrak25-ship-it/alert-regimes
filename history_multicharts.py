"""Seven independent Matplotlib daily panels for one month.

Only previously aggregated daily rows are visualized. This module never changes
features, models, forecasts or regime labels. Render to plain PNG bytes, avoiding
Streamlit chart state, browser JavaScript and the SVG sanitization issue.
"""
from __future__ import annotations

import calendar
from io import BytesIO

import numpy as np
import pandas as pd

from dashboard_data import REGION_UK


def prepare_month_comparison(daily: pd.DataFrame, regions: list[str], year: int, month: int) -> pd.DataFrame:
    """Return the exact daily records available in the requested month and regions.

    No missing day or source interval is imputed. Every row is checked before
    plotting; zero hours stay zero; duplicates or invalid hours fail closed.
    """
    if not regions or len(regions) != len(set(regions)):
        raise ValueError("Потрібні неповторні області для порівняння.")
    year, month = int(year), int(month)
    calendar.monthrange(year, month)
    required = {"region", "date", "recorded_hours", "has_source_interval"}
    if not required.issubset(daily):
        raise ValueError("Не вистачає полів добової історії.")
    part = daily.loc[daily["region"].isin(regions), list(required)].copy()
    part["date"] = pd.to_datetime(part["date"], errors="raise")
    part = part.loc[part["date"].dt.year.eq(year) & part["date"].dt.month.eq(month)].copy()
    if part.duplicated(["region", "date"]).any():
        raise ValueError("Повторені дати для області.")
    part["recorded_hours"] = pd.to_numeric(part["recorded_hours"], errors="coerce")
    if not np.isfinite(part["recorded_hours"]).all() or not part["recorded_hours"].between(0, 24.00001).all():
        raise ValueError("Тривалість за добу поза межами 0–24 год.")
    part["day"] = part["date"].dt.day
    return part.sort_values(["region", "date"]).reset_index(drop=True)


def comparison_table(part: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Reproducible numerical fallback when Windows blocks Matplotlib DLLs."""
    output = []
    for r in regions:
        one = part.loc[part["region"].eq(r)]
        output.append({
            "Область": REGION_UK.get(r, r),
            "Днів із добовими даними": len(one),
            "Середня тривалість, год": round(float(one["recorded_hours"].mean()), 2) if len(one) else None,
            "Днів із записом у джерелі": int(one["has_source_interval"].sum()) if len(one) else 0,
        })
    return pd.DataFrame(output)


def daily_wide_table(part: pd.DataFrame, regions: list[str], year: int, month: int) -> pd.DataFrame:
    """All calendar days across columns, preserving NaN for missing records."""
    days = calendar.monthrange(int(year), int(month))[1]
    matrix = part.pivot(index="region", columns="day", values="recorded_hours")
    matrix = matrix.reindex(index=regions, columns=range(1, days + 1))
    matrix.index = [REGION_UK.get(r, r) for r in matrix.index]
    matrix.columns = [f"{d:02d}" for d in matrix.columns]
    return matrix


def render_month_png(part: pd.DataFrame, regions: list[str], year: int, month: int) -> bytes:
    """Render one PNG containing seven clearly separated mini-bar charts.

    Uses the Agg raster backend directly (no GUI, no pyplot). PNG bytes update
    deterministically with the input month/region values, unlike embedded SVG.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    year, month = int(year), int(month)
    total_days = calendar.monthrange(year, month)[1]
    maximum = float(part["recorded_hours"].max()) if len(part) else 0.0
    ymax = max(2, 2 * int(np.ceil(maximum / 2)))
    fig = Figure(figsize=(12.4, max(7.0, len(regions) * 2.02)), dpi=110, facecolor="white")
    axes = fig.subplots(len(regions), 1, squeeze=False).ravel()
    fig.subplots_adjust(left=0.053, right=0.985, bottom=0.035, top=0.960, hspace=0.74)
    for ax, region in zip(axes, regions):
        group = part.loc[part["region"].eq(region)]
        ax.set_facecolor("white")
        if len(group):
            ax.bar(group["day"].astype(int), group["recorded_hours"], width=0.79,
                   color="#3478BD", edgecolor="#2D659B", linewidth=0.35, zorder=3)
        ax.set_xlim(0.3, total_days + 0.7)
        ax.set_ylim(0, ymax * 1.03)
        ax.set_xticks(range(1, total_days + 1))
        ax.set_xticklabels([f"{n:02d}" for n in range(1, total_days + 1)], fontsize=7)
        ax.set_yticks([0, ymax / 2, ymax])
        ax.tick_params(axis="y", labelsize=8, length=0, pad=5)
        ax.tick_params(axis="x", length=0, pad=6)
        ax.grid(axis="y", color="#E6EDF5", linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        mean = f"{group['recorded_hours'].mean():.2f} год/добу" if len(group) else "Немає добових рядків"
        ax.set_title(REGION_UK.get(region, region), loc="left", pad=9, fontsize=11, fontweight="bold", color="#183553")
        ax.text(1.0, 1.12, f"{len(group)}/{total_days} днів · {mean}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#64748B")
        ax.set_ylabel("год", fontsize=8, color="#64748B", rotation=0, labelpad=14)
    output = BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return output.getvalue()
