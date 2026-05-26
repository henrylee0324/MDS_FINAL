"""
Step C (optional): Multi-period knapsack — PDF §4.2 with x_ik (song i in period k).

Usage:
  python run_knapsack_multperiod.py
"""
from __future__ import annotations

import sys

import pandas as pd

from knapsack_solver import MultiPeriodParams, solve_multi_period
from paths import MULTIPERIOD_REPORT_TXT, MULTIPERIOD_SOLUTION_CSV, SCORES_CSV

PARAMS = MultiPeriodParams(
    periods=['Q1_春夏', 'Q2_秋冬'],
    budget_by_period={'Q1_春夏': 45.0, 'Q2_秋冬': 50.0},
    l_min_by_period={'Q1_春夏': 1, 'Q2_秋冬': 1},
    u_max_by_period={'Q1_春夏': 4, 'Q2_秋冬': 4},
    # 啟用時請依候選曲 loudness/tempo 調整，過嚴會 Infeasible
    loudness_min_by_period={},
    tempo_min_by_period={},
)


def main() -> None:
    if not SCORES_CSV.exists():
        print(f'Run score_synthetic_songs.py first. Missing {SCORES_CSV}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SCORES_CSV)
    sol_df, status = solve_multi_period(df, PARAMS)
    if status != 'Optimal':
        print(f'Solver: {status}. Relax period budgets/limits.', file=sys.stderr)
        sys.exit(1)

    sol_df.to_csv(MULTIPERIOD_SOLUTION_CSV, index=False)

    lines = [
        '=== 背包決策（多檔期 x_ik）===',
        f'periods={PARAMS.periods}  status={status}',
        '',
    ]
    for k in PARAMS.periods:
        sub = sol_df[sol_df['period'] == k]
        lines.append(f'【{k}】')
        if sub.empty:
            lines.append('  (無選曲)')
        else:
            for _, r in sub.sort_values('y_hat', ascending=False).iterrows():
                lines.append(
                    f"  {r['new_song_id']}  {r['profile_name']}  y_hat={r['y_hat']:.4f}  cost={r['cost']:.0f}"
                )
            lines.append(f"  小計 y_hat={sub['y_hat'].sum():.4f}  cost={sub['cost'].sum():.0f}  n={len(sub)}")
    lines.append(f"\n全期 y_hat={sol_df['y_hat'].sum():.4f}  cost={sol_df['cost'].sum():.0f}")

    report = '\n'.join(lines)
    MULTIPERIOD_REPORT_TXT.write_text(report, encoding='utf-8')
    print(report)
    print(f'\nSaved -> {MULTIPERIOD_SOLUTION_CSV}\n       -> {MULTIPERIOD_REPORT_TXT}')


if __name__ == '__main__':
    main()
