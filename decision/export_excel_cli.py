#!/usr/bin/env python3
"""僅從既有 outputs/ CSV 重新產生 Excel（不必重跑模型）。"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict

import pandas as pd

from export_excel import export_strategy_workbook
from knapsack_solver import KnapsackParams, greedy_top_baseline
from paths import (
    KNAPSACK_SOLUTION_CSV,
    OUTPUT_DIR,
    SCORES_CSV,
    SENSITIVITY_CSV,
    STRATEGY_SUMMARY_CSV,
    resolve_head200_csv,
)


def main() -> None:
    for p in (SCORES_CSV, STRATEGY_SUMMARY_CSV):
        if not p.exists():
            print(f'缺少 {p}，請先執行 bash run.sh', file=sys.stderr)
            sys.exit(1)

    summary = pd.read_csv(STRATEGY_SUMMARY_CSV).iloc[0].to_dict()
    scores = pd.read_csv(SCORES_CSV)
    full = pd.read_csv(KNAPSACK_SOLUTION_CSV) if KNAPSACK_SOLUTION_CSV.exists() else scores
    sens = pd.read_csv(SENSITIVITY_CSV) if SENSITIVITY_CSV.exists() else pd.DataFrame()

    chosen = summary['selected_ids'].split(',')
    ranked = full.sort_values('y_hat', ascending=False)
    budget = float(summary['budget'])
    baseline = greedy_top_baseline(scores, budget)

    meta_path = OUTPUT_DIR / 'strategy_meta.json'
    profile, cfg = {}, {'base_row': int(summary.get('base_row', 0)), 'budget': budget}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        profile = meta.get('profile', profile)
        cfg = meta.get('config', cfg)

    kparams = KnapsackParams(
        budget=budget,
        l_min=int(summary.get('l_min', 3)),
        u_max=int(summary.get('u_max', 5)),
    )

    path = export_strategy_workbook(
        csv_path=resolve_head200_csv(),
        profile=profile,
        cfg=cfg,
        ranked=ranked,
        chosen=chosen,
        kparams=asdict(kparams),
        summary_row=summary,
        baseline=baseline,
        sensitivity=sens,
        full_solution=full,
    )
    print(f'Excel -> {path}')


if __name__ == '__main__':
    main()
