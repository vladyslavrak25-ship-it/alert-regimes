"""Run exploratory tests on earlier windows; NEVER replace saved 12–18 Oct forecast."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from model_robustness import FIRST_ORIGINS, TARGET_REGIONS, run_exploration
from regimes import read_daily_table

ROOT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Earlier out-of-sample model comparison (exploratory)')
    parser.add_argument('--quick', action='store_true',
                        help='One fold per region for initial checking; not enough for conclusions.')
    args = parser.parse_args()
    forecast = ROOT / 'outputs' / 'week_2026-10-12_to_18.csv'
    choice = ROOT / 'outputs' / 'model_selection.csv'
    before = {name: digest(path) for name, path in [('forecast',forecast),('selection',choice)]}
    daily = read_daily_table(ROOT / 'outputs' / 'daily_alert_hours.csv')
    origins = FIRST_ORIGINS[:1] if args.quick else FIRST_ORIGINS
    predictions, folds, summary = run_exploration(daily, TARGET_REGIONS, origins)
    after = {name: digest(path) for name, path in [('forecast',forecast),('selection',choice)]}
    if before != after:
        raise RuntimeError('Файли обраного прогнозу несподівано змінилися.')
    folder = ROOT / 'analysis'
    folder.mkdir(exist_ok=True)
    predictions.to_csv(folder/'exploratory_predictions.csv', index=False, encoding='utf-8-sig')
    folds.to_csv(folder/'exploratory_folds.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(folder/'exploratory_summary.csv', index=False, encoding='utf-8-sig')
    print('\nДОДАТКОВИЙ ЕКСПЕРИМЕНТ (без зміни фінальної моделі)')
    print(f'Регіонів: {len(TARGET_REGIONS)}; вікон на регіон: {len(origins)}')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nЗбережено лише analysis/exploratory_*.csv')
    print('Фінальні model_selection.csv та week_2026-10-12_to_18.csv НЕ змінено (SHA-256 збігається).')
    print('Це дослідницьке порівняння, не новий незалежний фінальний тест.')


if __name__ == '__main__':
    main()
