"""
Integer programming knapsack (PDF §4.2) — full constraints.

  (1) max Σ y_hat * x
  (2) Σ cost <= C_k  per period
  (3)(4) Σ a_ij x >= T_jk Σ x  per period per acoustic metric
  (5) L_k <= Σ x <= U_k per period
  (6) Σ_k x_ik <= 1 per song
  (7) x binary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

try:
    import pulp
except ImportError as e:
    raise ImportError('pip install pulp') from e


@dataclass
class KnapsackParams:
    budget: float = 90.0
    l_min: int = 3
    u_max: int = 5
    loudness_min: Optional[float] = None
    tempo_min: Optional[float] = None


@dataclass
class MultiPeriodParams:
    periods: list[str] = field(default_factory=lambda: ['Q1', 'Q2'])
    budget_by_period: dict[str, float] = field(default_factory=dict)
    l_min_by_period: dict[str, int] = field(default_factory=dict)
    u_max_by_period: dict[str, int] = field(default_factory=dict)
    # period -> metric -> minimum average (T_jk); None 跳過
    acoustic_min_by_period: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    cost_multiplier: float = 1.0


def _row(df: pd.DataFrame, song_id: str) -> pd.Series:
    return df.loc[df['new_song_id'] == song_id].iloc[0]


def solve_single_period(df: pd.DataFrame, params: KnapsackParams) -> tuple[list[str], str]:
    songs = df['new_song_id'].tolist()
    prob = pulp.LpProblem('HitSong_Knapsack_Single', pulp.LpMaximize)
    x = {i: pulp.LpVariable(f'x_{i}', cat='Binary') for i in songs}

    prob += pulp.lpSum(_row(df, i)['y_hat'] * x[i] for i in songs)
    prob += pulp.lpSum(_row(df, i)['cost'] * x[i] for i in songs) <= params.budget
    prob += pulp.lpSum(x[i] for i in songs) >= params.l_min
    prob += pulp.lpSum(x[i] for i in songs) <= params.u_max

    for metric, t in [('loudness', params.loudness_min), ('tempo', params.tempo_min)]:
        if t is not None and metric in df.columns:
            prob += (
                pulp.lpSum(float(_row(df, i).get(metric, 0) or 0) * x[i] for i in songs)
                >= t * pulp.lpSum(x[i] for i in songs)
            )

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    chosen = [i for i in songs if pulp.value(x[i]) > 0.5]
    return chosen, status


def solve_multi_period(df: pd.DataFrame, params: MultiPeriodParams) -> tuple[pd.DataFrame, str]:
    songs = df['new_song_id'].tolist()
    periods = params.periods

    prob = pulp.LpProblem('HitSong_Knapsack_MultiPeriod', pulp.LpMaximize)
    x = {(i, k): pulp.LpVariable(f'x_{i}_{k}', cat='Binary') for i in songs for k in periods}

    # (1) objective
    prob += pulp.lpSum(_row(df, i)['y_hat'] * x[i, k] for i in songs for k in periods)

    for k in periods:
        # (2) budget
        ck = params.budget_by_period.get(k, 50.0)
        prob += pulp.lpSum(
            _row(df, i)['cost'] * params.cost_multiplier * x[i, k] for i in songs
        ) <= ck

        # (5) release volume
        lk = params.l_min_by_period.get(k, 1)
        uk = params.u_max_by_period.get(k, len(songs))
        prob += pulp.lpSum(x[i, k] for i in songs) >= lk
        prob += pulp.lpSum(x[i, k] for i in songs) <= uk

        # (3)(4) acoustic average floors per metric
        thresh = params.acoustic_min_by_period.get(k, {}) or {}
        for metric, t in thresh.items():
            if t is None or metric not in df.columns:
                continue
            prob += (
                pulp.lpSum(float(_row(df, i).get(metric, 0) or 0) * x[i, k] for i in songs)
                >= float(t) * pulp.lpSum(x[i, k] for i in songs)
            )

    # (6) each song at most once
    for i in songs:
        prob += pulp.lpSum(x[i, k] for k in periods) <= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]

    rows = []
    for i in songs:
        for k in periods:
            if pulp.value(x[i, k]) > 0.5:
                r = _row(df, i).to_dict()
                r['period'] = k
                r['selected'] = 1
                rows.append(r)
    return pd.DataFrame(rows), status


def greedy_top_baseline(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    ordered = df.sort_values('y_hat', ascending=False)
    picked, spent = [], 0.0
    for _, r in ordered.iterrows():
        if spent + r['cost'] <= budget:
            picked.append(r['new_song_id'])
            spent += float(r['cost'])
    return df[df['new_song_id'].isin(picked)].copy()


def traditional_multperiod_greedy(
    df: pd.DataFrame,
    params: MultiPeriodParams,
) -> pd.DataFrame:
    """
    傳統 A&R：各檔期獨立依 ŷ 貪婪填滿預算，已選歌曲不重複。
    不保證滿足聲學平均門檻。
    """
    songs = df.sort_values('y_hat', ascending=False)['new_song_id'].tolist()
    used: set[str] = set()
    rows = []

    for k in params.periods:
        budget = params.budget_by_period.get(k, 50.0)
        uk = params.u_max_by_period.get(k, len(songs))
        lk = params.l_min_by_period.get(k, 1)
        spent = 0.0
        picked: list[str] = []

        for sid in songs:
            if sid in used:
                continue
            cost = float(df.loc[df['new_song_id'] == sid, 'cost'].iloc[0])
            if spent + cost <= budget and len(picked) < uk:
                picked.append(sid)
                spent += cost
            if spent >= budget * 0.98 and len(picked) >= lk:
                break

        # 若未達 L_k，強制補最低成本未使用曲
        for sid in songs:
            if len(picked) >= lk:
                break
            if sid in used:
                continue
            cost = float(df.loc[df['new_song_id'] == sid, 'cost'].iloc[0])
            if spent + cost <= budget:
                picked.append(sid)
                spent += cost

        for sid in picked:
            used.add(sid)
            r = _row(df, sid).to_dict()
            r['period'] = k
            r['selected'] = 1
            rows.append(r)

    return pd.DataFrame(rows)


def check_acoustic_constraints(schedule: pd.DataFrame, params: MultiPeriodParams) -> pd.DataFrame:
    """回傳各檔期各指標的平均值 vs 門檻（供報告）。"""
    rows = []
    if schedule.empty:
        return pd.DataFrame()
    for k in params.periods:
        sub = schedule[schedule['period'] == k]
        if sub.empty:
            continue
        n = len(sub)
        thresh = params.acoustic_min_by_period.get(k, {}) or {}
        for metric in set(list(thresh.keys()) + ['loudness', 'tempo', 'danceability', 'energy']):
            if metric not in sub.columns:
                continue
            avg = float(sub[metric].mean())
            t = thresh.get(metric)
            rows.append({
                'period': k,
                'metric': metric,
                'average': avg,
                'threshold': t,
                'pass': (t is None) or (avg >= float(t) - 1e-9),
                'n_songs': n,
            })
    return pd.DataFrame(rows)
