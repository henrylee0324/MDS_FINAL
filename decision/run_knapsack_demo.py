"""
Step C: Single-period knapsack (choose subset of 10 synthetic songs).

Usage:
  python run_knapsack_demo.py
"""
from __future__ import annotations

import sys

import pandas as pd

from knapsack_solver import KnapsackParams, greedy_top_baseline, solve_single_period
from paths import KNAPSACK_REPORT_TXT, KNAPSACK_SOLUTION_CSV, SCORES_CSV

PARAMS = KnapsackParams(
    budget=90.0,
    l_min=3,
    u_max=5,
    loudness_min=None,
    tempo_min=None,
)


def main() -> None:
    if not SCORES_CSV.exists():
        print(f'Run score_synthetic_songs.py first. Missing {SCORES_CSV}', file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SCORES_CSV)
    chosen, status = solve_single_period(df, PARAMS)
    if status != 'Optimal':
        print(f'Solver: {status}. Relax budget or L_MIN/U_MAX in run_knapsack_demo.py', file=sys.stderr)
        sys.exit(1)

    sol = df[df['new_song_id'].isin(chosen)].copy()
    baseline = greedy_top_baseline(df, PARAMS.budget)

    full = df.copy()
    full['selected'] = full['new_song_id'].isin(chosen).astype(int)
    full.to_csv(KNAPSACK_SOLUTION_CSV, index=False)

    lines = [
        '=== 背包決策（單檔期）===',
        f'BUDGET={PARAMS.budget}  L_MIN={PARAMS.l_min}  U_MAX={PARAMS.u_max}  status={status}',
        '',
        '【最優解】',
    ]
    for _, r in sol.sort_values('y_hat', ascending=False).iterrows():
        lines.append(f"  {r['new_song_id']}  {r['profile_name']}  y_hat={r['y_hat']:.4f}  cost={r['cost']:.0f}  {r['tier']}")
    lines += [
        f"  合計 y_hat={sol['y_hat'].sum():.4f}  cost={sol['cost'].sum():.0f}  n={len(sol)}",
        '',
        '【對照：TOP 貪婪直到預算滿】',
    ]
    for _, r in baseline.sort_values('y_hat', ascending=False).iterrows():
        lines.append(f"  {r['new_song_id']}  y_hat={r['y_hat']:.4f}  cost={r['cost']:.0f}")
    lines += [
        f"  合計 y_hat={baseline['y_hat'].sum():.4f}  cost={baseline['cost'].sum():.0f}  n={len(baseline)}",
        f"  Δy_hat = {sol['y_hat'].sum() - baseline['y_hat'].sum():.4f}",
    ]
    report = '\n'.join(lines)
    KNAPSACK_REPORT_TXT.write_text(report, encoding='utf-8')
    print(report)
    print(f'\nSaved -> {KNAPSACK_SOLUTION_CSV}\n       -> {KNAPSACK_REPORT_TXT}')


if __name__ == '__main__':
    main()
