"""Alert Regimes: transparent research portal for precomputed stage 1–3 artifacts.

Run: python -m streamlit run app.py
This is NOT a live alert, safety, or attack forecast system.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_data import DashboardDataError, MODEL_UK, REGION_UK, REGIME_UK, load_outputs
from dashboard_auto import AutoForecastError, load_latest_week
from history_heatmap import MONTH_UK, month_summary, year_heatmap, month_details
from history_daily_chart import region_chart_payload
from dashboard_analysis import (
    current_snapshot_warning, regional_category_counts, selected_day_forecast,
    subset_all_regions, verify_frozen_windows, verify_model_choices,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CATS = ("low", "medium", "high")
CAT_LABELS = {"Низька": "low", "Помірна": "medium", "Висока": "high"}
DAYS = [date(2026, 10, 12 + i) for i in range(7)]
DAYS_UK = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]

st.set_page_config(page_title="Alert Regimes | Дослідницький проєкт", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2.0rem; padding-bottom: 3rem;}
h1 {letter-spacing: -0.04em;}
[data-testid="stMetric"] {background: rgba(128,128,128,.065); border-radius: 12px; padding: 12px;}
[data-testid="stAlert"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

try:
    data = load_outputs(OUT)
    verify_frozen_windows(data["selection"], data["thresholds"], data["manifest"])
    model_counts = verify_model_choices(data["future"], data["selection"])
except (DashboardDataError, ValueError, OSError, KeyError, TypeError) as exc:
    st.error(f"Не вдалося перевірити результати: {exc}")
    st.info("Переконайся, що в outputs лежать файли з одного узгодженого запуску етапів 1–3. Не змішуй старі та нові CSV.")
    st.stop()

regions = list(data["selection"]["region"])
cutoff = pd.Timestamp(data["manifest"]["date_of_last_complete_observation"]).date()
full_history = data["daily"].loc[
    data["daily"]["region"].isin(regions)
    & (data["daily"]["date"].dt.date <= cutoff)
].copy()
start_available = full_history["date"].min().date()
end_available = full_history["date"].max().date()

if "screen" not in st.session_state:
    st.session_state.screen = "intro"
if "forecast_day" not in st.session_state:
    st.session_state.forecast_day = 0


def disclaimer():
    st.warning(
        "Це дослідницька оцінка **категорій зафіксованої тривалості тривог**, "
        "а НЕ оцінка безпеки, ризику атаки чи система сповіщення. "
        "Навіть «низька» активність не означає безпечний день. Орієнтуйся на офіційні сповіщення."
    )


def intro():
    st.caption("KSE Agentic School · Python Stage 2 · Аналітичний проєкт")
    st.title("Alert Regimes")
    st.subheader("Історична активність повітряних тривог та ймовірнісний прогноз для семи областей")
    st.markdown(
        "Проєкт досліджує, скільки годин тривоги **зафіксовано** протягом доби у "
        "волонтерських даних, і відносить дні до **низької, помірної або високої** "
        "активності. Для 12–18 жовтня 2026 року показує ймовірність кожної категорії "
        "для семи областей. Історичний огляд доступний з 2022 року."
    )
    disclaimer()
    st.subheader("Чому саме сім областей?")
    st.markdown(
        "Ми свідомо обмежили практичну задачу сімома кандидатами, щоб перевірити "
        "якість календарних даних, стабільність порогів і прогнозну якість **для кожного регіону**. "
        "Кандидатів було визначено до перегляду фінального тесту, а придатність перевірено "
        "за **180 навчальними днями (16.03–11.09.2026)**: не менше 175 днів із записами, "
        "середня тривалість ≥3 год/добу, частка умовних годин ≤10%, "
        "розрізнювані пороги та достатньо прикладів усіх трьох категорій."
    )
    st.info(
        "Ці області **не є рейтингом найнебезпечніших**. Решта регіонів не є менш важливими; "
        "ми обрали сфокусовану, відтворювану задачу. Менша кількість областей сама по собі "
        "не гарантує математичної точності — її перевіряємо окремо за Brier score."
    )
    readiness_path = OUT / "region_readiness.csv"
    if readiness_path.exists():
        readiness = pd.read_csv(readiness_path, encoding="utf-8-sig")
        readiness = readiness.loc[readiness["region"].isin(regions)].copy()
        if not readiness.empty:
            readiness["Область"] = readiness["region"].map(REGION_UK)
            readiness["Частка умовних годин, %"] = readiness["naive_hours_share"] * 100
            with st.expander("Показати критерії відбору для кожної області"):
                st.dataframe(readiness[["Область", "train_days", "days_with_source_interval",
                    "mean_hours", "Частка умовних годин, %", "low_days", "medium_days", "high_days"]]
                    .rename(columns={"train_days":"Навчальних днів",
                        "days_with_source_interval":"Днів із записами", "mean_hours":"Год/добу",
                        "low_days":"Низьких", "medium_days":"Помірних", "high_days":"Високих"}),
                    hide_index=True, use_container_width=True)
    st.subheader("Як читати результати")
    st.markdown(
        "- **Історія** — усі наявні календарні дні з 2022 року, із можливістю звузити період. "
        "Нуль годин у CSV не доводить відсутності реальних тривог.\n"
        "- **Категорії** — індивідуальні для кожної області пороги, обчислені за 180 днями "
        "навчання. Для давніших років це *ретроспективна класифікація за порогами 2026 року*.\n"
        "- **Прогноз** — натисни на день тижня, обери категорію, порівняй усі сім областей.\n"
        "- **Методологія** — окремий вибір моделі для кожної області, валідація, тест і обмеження."
    )
    st.caption(current_snapshot_warning(data["audit"], data["manifest"]))
    if st.button("Розпочати дослідження →", type="primary", use_container_width=True):
        st.session_state.screen = "analysis"
        st.rerun()


def period_selector(key: str):
    period = st.date_input(
        "Період історичного порівняння (за замовчуванням — уся історія)",
        value=(start_available, end_available), min_value=start_available,
        max_value=end_available, key=key,
    )
    if len(period) != 2:
        st.info("Оберіть обидві дати для порівняння.")
        st.stop()
    return period[0], period[1]


@st.cache_data(show_spinner=False)
def cached_months(history: pd.DataFrame, regions_tuple: tuple[str, ...]) -> pd.DataFrame:
    return month_summary(history, list(regions_tuple))


def history_page():
    st.header("Історія · теплова карта активності за місяцями")
    st.markdown(
        "**Сім областей × 12 місяців.** Кожна клітинка — середня кількість "
        "**зафіксованих годин тривог за календарну добу** в цьому місяці. "
        "Це не число тривог і не рівень безпеки. Усі роки з 2022-го доступні окремо."
    )
    monthly = cached_months(full_history, tuple(regions))
    years = sorted(monthly['year'].unique().tolist(), reverse=True)
    chosen_year = st.selectbox('Рік', years, key='history_heatmap_year')
    matrix, coverage = year_heatmap(monthly, regions, int(chosen_year))
    # One scale for all years: selecting a year doesn't recolor the same value.
    max_scale = max(float(monthly['mean_hours'].max()), 1.0)
    st.caption(
        f"Колір показує середню записану тривалість: від 0 до {max_scale:.1f} год/добу "
        "(однакова шкала для всіх років). Порожня клітинка означає відсутні "
        "дані за цей місяць, а не нуль тривог."
    )
    visual = matrix.style.format('{:.1f}', na_rep='—').background_gradient(
        cmap='Blues', axis=None, vmin=0, vmax=max_scale,
    )
    st.dataframe(visual, use_container_width=True, height=305)
    st.caption(
        f"Показано {int(coverage.fillna(0).to_numpy().sum()):,} календарних "
        f"рядків за {chosen_year} рік для семи областей. "
        "Неповні місяці також відображаються — перевір їхню кількість днів нижче."
    )
    with st.expander('Скільки днів містить кожна клітинка?'):
        st.dataframe(coverage.style.format('{:.0f}', na_rep='—'),
                     use_container_width=True, height=305)
        st.caption('Для повного місяця — 28–31 день; для неповного — менше. '
                   'Початок історії у 2022 році та жовтень 2026 можуть бути неповними.')

    st.divider()
    st.subheader('Деталізація · одна область і один місяць')
    st.caption('Кожен стовпчик — записана тривалість тривог за цілу календарну добу. '
               'Внизу показано всі числа місяця, без годинних позначок 06 AM / 12 PM.')
    active_months = sorted(monthly.loc[monthly['year'] == chosen_year, 'month'].unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        detail_month = st.selectbox('Місяць', active_months,
                                    format_func=lambda m: MONTH_UK[m - 1],
                                    key='history_detail_month')
    with c2:
        available_regions = monthly.loc[
            (monthly['year'] == chosen_year) & (monthly['month'] == detail_month), 'region'
        ].tolist()
        ordered_available = [r for r in regions if r in available_regions]
        detail_region = st.selectbox('Область для деталізації', ordered_available,
                                     format_func=lambda r: REGION_UK.get(r, r),
                                     key='history_detail_region')
    details = month_details(full_history, detail_region, int(chosen_year), int(detail_month))
    # Ordinal day labels 01..31 eliminate timestamp zoom labels (06 AM, 12 PM).
    # Vega-Lite bar widths are fixed at 22px, with a 32px step for readability.
    chart_rows, daily_spec, chart_key = region_chart_payload(
        details, int(chosen_year), int(detail_month), detail_region
    )
    st.caption(
        f"Показано: **{REGION_UK.get(detail_region, detail_region)}** · "
        f"{MONTH_UK[int(detail_month) - 1]} {int(chosen_year)}"
    )
    st.vega_lite_chart(
        data=chart_rows, spec=daily_spec, key=chart_key, use_container_width=False
    )
    if len(details) < len(daily_spec['encoding']['x']['scale']['domain']):
        st.caption('У вибраному місяці є дні без добового рядка. На графіку вони залишені '
                   'порожніми, а не замінені нульовою тривалістю.')
    c1, c2, c3 = st.columns(3)
    c1.metric('Днів із добовими даними', len(details))
    c2.metric('Середня тривалість за день', f"{details['recorded_hours'].mean():.2f} год")
    c3.metric('Днів із записом у джерелі', int(details['has_source_interval'].sum()))
    with st.expander('Переглянути добові значення та умовні завершення'):
        view = details.rename(columns={
            'date': 'Дата', 'recorded_hours': 'Записані години',
            'has_source_interval': 'Є запис у джерелі',
            'naive_added_hours': 'Умовно завершені години',
        })
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption('Нуль годин означає відсутність записаного інтервалу, '
                   'а не доведену відсутність реальної тривоги. `naive=True` '
                   'враховано, але внесок збережено окремо.')


def category_page():
    st.header("Рівні активності · порівняння регіонів")
    st.caption("Оберіть категорію, а не окрему область. Пороги зафіксовані індивідуально "
               "за навчальними даними 16.03–11.09.2026.")
    start, end = period_selector("category_period_v2")
    part = subset_all_regions(full_history, regions, start, end, cutoff)
    counts = regional_category_counts(part, data["thresholds"], regions)
    category_name = st.radio("Яку категорію порівнюємо?", list(CAT_LABELS),
                             index=2, horizontal=True, key="category_radio_v2")
    count_col, ratio_col = st.columns(2)
    with count_col:
        st.metric(f"{category_name} · днів у 7 областях", f"{counts[category_name].sum():,}")
    with ratio_col:
        st.metric("Календарних рядків у вибірці", f"{counts['Усього днів'].sum():,}")
    chart = counts[["Область", category_name]].set_index("Область").sort_values(category_name, ascending=False)
    st.bar_chart(chart, horizontal=True, height=320)
    counts["Частка вибраної категорії, %"] = (100*counts[category_name]/counts["Усього днів"]).round(1)
    st.subheader("Усі три категорії та контроль сум")
    st.dataframe(counts[["Область", "Низька", "Помірна", "Висока", "Усього днів",
                         "Частка вибраної категорії, %", "Поріг q33, год", "Поріг q75, год"]].round(2),
                 hide_index=True, use_container_width=True)
    total_classified = counts[["Низька", "Помірна", "Висока"]].to_numpy().sum()
    st.success(f"Перевірка: {int(total_classified):,} днів у трьох категоріях = "
               f"{int(counts['Усього днів'].sum()):,} календарних рядків.")
    st.info("Для 2022–2025 років застосовано пороги, обчислені на навчальному вікні "
            "2026 року. Це порівняльна ретроспективна розмітка, а не історичний "
            "прогноз чи офіційний рівень тривоги. Один і той самий час може "
            "потрапити в різні категорії у різних областях.")


def forecast_page():
    st.header("Прогноз на тиждень · обери день")
    forecast_table = data["future"]
    forecast_days = DAYS
    auto = None
    try:
        auto = load_latest_week(OUT, dict(zip(data["selection"]["region"], data["selection"]["selected_model"])))
    except AutoForecastError as exc:
        st.warning(f"Автоматичний прогноз недоступний: {exc}. Показано перевірений архів.")
    if auto is not None:
        mode = st.radio("Версія прогнозу", ["Останній автоматичний", "Архівний · 12–18 жовтня"], horizontal=True)
        if mode == "Останній автоматичний":
            forecast_table, meta = auto
            start = date.fromisoformat(meta["forecast_first_day"])
            forecast_days = [start + pd.Timedelta(days=i) for i in range(7)]
            st.caption(f"Автоматичний зріз: {meta['last_training_day']}; прогноз: {meta['forecast_first_day']} — {meta['forecast_last_day']}.")
            st.caption("Час публікації кожного запису в джерелі не відомий; ця перевірка використовує консервативну межу.")
        else:
            st.caption(current_snapshot_warning(data["audit"], data["manifest"]))
    else:
        st.caption(current_snapshot_warning(data["audit"], data["manifest"]))
        st.caption("Після першого успішного GitHub Actions тут можна буде перемикати автоматичний і архівний прогнози.")
    st.info("Ймовірності показують лише відносну категорію записаних годин. "
            "Вони НЕ означають імовірність атаки або безпеки у вибраний день.")
    cols = st.columns(7, gap="small")
    for i, (col, day) in enumerate(zip(cols, forecast_days)):
        with col:
            if st.button(f"{DAYS_UK[i]} · {day:%d.%m}", key=f"weekday_{i}",
                         type="primary" if st.session_state.forecast_day == i else "secondary",
                         use_container_width=True):
                st.session_state.forecast_day = i
                st.rerun()
    day = forecast_days[st.session_state.forecast_day]
    st.subheader(f"{DAYS_UK[st.session_state.forecast_day]}, {day:%d.%m.%Y}")
    chosen = selected_day_forecast(forecast_table, day, regions)
    cat = st.radio("Яку категорію порівнюємо?", list(CAT_LABELS),
                   index=2, horizontal=True, key="forecast_category_v2")
    chart = chosen[["Область", cat]].sort_values(cat, ascending=False).set_index("Область")
    st.bar_chart(chart, horizontal=True, height=330)
    st.caption(f"Показано **ймовірність категорії «{cat.lower()}»** для кожної області, "
               "а не спільний рейтинг небезпеки регіонів.")
    st.subheader("Всі ймовірності на вибрану дату")
    formatted = chosen[["Область", "Низька", "Помірна", "Висока", "selected_model", "horizon"]].copy()
    formatted["Модель"] = formatted["selected_model"].map(MODEL_UK)
    for cat_label in CAT_LABELS:
        formatted[cat_label] = formatted[cat_label].map(lambda x: f"{x:.1f}%")
    formatted = formatted.rename(columns={"horizon": "Горизонт, днів"})
    st.dataframe(formatted[["Область", "Низька", "Помірна", "Висока", "Модель", "Горизонт, днів"]],
                 hide_index=True, use_container_width=True)
    with st.expander("Чому для Чернігівської та деяких інших областей усі дні однакові?", expanded=True):
        st.write("Для Чернігівської, Харківської, Сумської, Запорізької та Дніпропетровської "
                 "на попередній валідації обрано **Baseline**. Він бере частоти трьох категорій "
                 "за **останні 28 доступних днів** та застосовує їх до всіх майбутніх дат із "
                 "цього самого зрізу; не використовує день тижня. Тому однакові ймовірності "
                 "протягом тижня **очікувані**, а не помилка відображення. "
                 "Полтавська та Миколаївська використовують Logistic Regression, "
                 "яка враховує календарні ознаки й горизонт прогнозу.")
    st.caption(f"Область має рівно одну обрану для публікації модель: "
               f"{model_counts['baseline']} Baseline і {model_counts['logistic']} Logistic Regression. "
               "Це **не** одна глобальна модель для всіх семи областей: алгоритм вибрано "
               "окремо за валідацією.")


def methodology_page():
    st.header("Методологія, перевірка моделей і обмеження")
    disclaimer()
    st.subheader("Як вибирали алгоритм")
    st.markdown(
        "Ми **експериментально порівняли два кандидати**: Baseline (частоти останніх 28 днів) "
        "та багатокласову Logistic Regression. **Для прогнозу кожної області обрано рівно одну** "
        "за нижчим Brier score на валідації 15.08–11.09.2026. Інший кандидат лишається "
        "у файлах тільки для дослідницького порівняння. У різних областях переможці можуть різнитися."
    )
    selection = data["selection"].copy()
    selection["Область"] = selection["region"].map(REGION_UK)
    selection["Фінальна модель"] = selection["selected_model"].map(MODEL_UK)
    st.dataframe(selection[["Область", "Фінальна модель", "selection_gap_brier_logistic",
                             "selection_gap_brier_baseline", "selection_forecasts"]]
        .rename(columns={"selection_gap_brier_logistic": "Валідація: Logistic Brier",
                         "selection_gap_brier_baseline": "Валідація: Baseline Brier",
                         "selection_forecasts": "Валідаційних дат"}).round(4),
                 hide_index=True, use_container_width=True)
    st.caption("Brier score — сума трьох квадратів похибки ймовірностей (від 0 до 2); "
               "**менше — краще**. Це не процент правильних відповідей.")
    st.subheader("Перевірка відсутності витоку через пороги")
    st.markdown(
        "**Вибір моделі:** пороги для цього експерименту обчислені за 180 днями "
        "**до 14 серпня включно**; прогнози оцінені **15 серпня – 11 вересня**. "
        "Дані прогнозованих днів не використовувалися для визначення цих порогів.\n\n"
        "**Фінальний історичний тест:** остаточні пороги зафіксовані "
        "**16 березня – 11 вересня**; оцінка — **12 вересня – 9 жовтня**. "
        "На кожній історичній точці прогнозу модель навчалась лише на минулому "
        "щодо цієї точки. Відомі на момент прогнозу дні могли входити в наступне "
        "навчання — це rolling-origin підхід."
    )
    st.subheader("Оцінка на фінальному історичному тесті")
    metrics = data["metrics"].copy()
    metrics["Область"] = metrics["region"].map(REGION_UK)
    st.dataframe(metrics[["Область", "holdout_A_selected_brier", "holdout_B_selected_brier",
                          "holdout_B_baseline_brier", "holdout_B_logistic_brier"]]
        .rename(columns={"holdout_A_selected_brier": "Обрана: горизонти 1–7",
                         "holdout_B_selected_brier": "Обрана: горизонти 3–9",
                         "holdout_B_baseline_brier": "Baseline: 3–9",
                         "holdout_B_logistic_brier": "Logistic: 3–9"}).round(4),
                 hide_index=True, use_container_width=True)
    st.caption("Історичний експеримент A: 28 унікальних прогнозованих днів на область; "
               "B: 26. Їхні дати перетинаються, тому результати не можна вважати "
               "двома незалежними вибірками. Історичний holdout уже оглядався "
               "до виправлення протоколу: це скоригована, але не абсолютно незаймана перевірка.")
    st.subheader("Про повноту даних і свіжість")
    st.markdown(
        "- Використано Volunteer CSV (eTryvoga), а не повний офіційний журнал.\n"
        "- `naive=True` означає умовне 30-хвилинне завершення деяких записів.\n"
        "- 0 годин у щоденній таблиці означає *відсутність відповідного інтервалу в CSV*, "
        "а не підтверджену відсутність тривоги.\n"
        "- Час первинної публікації та пізніших правок окремих записів невідомий; "
        "ідеальної історичної доступності даних (*point-in-time*) довести не можемо.\n"
        "- Офіційні жовті/червоні рівні не визначаються з цих чотирьох полів.\n"
        "- Прогноз зафіксовано до появи фактичних значень прогнозованих днів. "
        "Архівний прогноз не перезаписується; нові результати публікуються окремо через GitHub Actions."
    )
    st.info(current_snapshot_warning(data["audit"], data["manifest"]))
    with st.expander("Технічні дані про знімок"):
        st.json({
            "last_complete_observation": str(cutoff),
            "source_latest_start": data["audit"].get("latest_start_kyiv"),
            "data_sha256": data["audit"].get("input_sha256"),
            "forecast_generated_at_kyiv": data["manifest"].get("created_at_kyiv"),
            "forecast_horizons": data["manifest"].get("forecast_horizons"),
        })


if st.session_state.screen == "intro":
    intro()
else:
    head_left, head_right = st.columns([4, 1])
    with head_left:
        st.title("Alert Regimes")
        st.caption("Дослідження зафіксованої активності · 7 областей · 2022–2026")
    with head_right:
        if st.button("← Про проєкт", use_container_width=True):
            st.session_state.screen = "intro"
            st.rerun()
    page = st.radio("Розділи дослідження", ["Історія", "Категорії", "Прогноз", "Методологія"],
                    horizontal=True, key="navigation_v2", label_visibility="collapsed")
    st.divider()
    if page == "Історія":
        history_page()
    elif page == "Категорії":
        category_page()
    elif page == "Прогноз":
        forecast_page()
    else:
        methodology_page()
    st.divider()
    st.caption(f"Alert Regimes · Джерело: Volunteer CSV / eTryvoga · остання включена доба: {cutoff} · "
               "дослідницькі результати, не система сповіщення")
