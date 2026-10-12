"""Smoke tests of seven-panel historical visualization."""
from io import BytesIO
from pathlib import Path
import pandas as pd
import pytest
from history_multicharts import prepare_month_comparison, render_month_png, daily_wide_table
from dashboard_data import load_outputs

def test_separate_region_values_and_missing_days():
    regions = ["Chernihivska oblast", "Zaporizka oblast"]
    frame = pd.DataFrame({
        "region": [regions[0], regions[1]],
        "date": ["2026-01-01", "2026-01-01"],
        "recorded_hours": [4.71, 6.81],
        "has_source_interval": [True, True],
    })
    month = prepare_month_comparison(frame, regions, 2026, 1)
    matrix = daily_wide_table(month, regions, 2026, 1)
    assert matrix.loc["Чернігівська область", "01"] == 4.71
    assert matrix.loc["Запорізька область", "01"] == 6.81
    assert pd.isna(matrix.loc["Запорізька область", "02"])

def test_real_png_is_renderable():
    image_module = pytest.importorskip("PIL.Image")
    root = Path(__file__).resolve().parents[1]
    data = load_outputs(root / "outputs")
    regions = data["selection"]["region"].tolist()
    assert len(regions) == 7
    month = prepare_month_comparison(data["daily"], regions, 2026, 1)
    assert len(month) == 217
    img = render_month_png(month, regions, 2026, 1)
    assert img.startswith(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
    with image_module.open(BytesIO(img)) as im:
        im.load()
        assert im.width > 900 and im.height > 1200

def test_app_uses_seven_panels_without_region_dropdown():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "st.image(png" in source
    assert "history_detail_region" not in source
    assert "_static_monthly_bars" not in source
