#!/usr/bin/env python3
"""
Hit Song Intelligence — 分析決策主程式

整合：資料診斷 → 20 首模擬新歌 → 熱度預測 → 多檔期 ILP → 傳統對照 → Excel

用法:
  cd MDS_FINAL-main/decision
  python strategy.py
  python strategy.py --csv ../../head200_MSD_with_all_features_categorical_encoded.csv
  python strategy.py --base-row 3 --budget 100 --multperiod
  python strategy.py --sensitivity-only   # 僅預算敏感度（需已有 song_scores.csv）
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config_columns import (
    BASE_ROW_INDEX,
    COST_TIERS,
    DEFAULT_ACOUSTIC_POLICY,
    DEFAULT_BUDGET_BY_PERIOD,
    DEFAULT_L_MIN_BY_PERIOD,
    DEFAULT_PERIODS,
    DEFAULT_U_MAX_BY_PERIOD,
    N_SYNTHETIC_SONGS,
    PROFILES,
)
from calibration import calibrate_acoustic_thresholds
from io_utils import feature_columns, find_target, load_head200
from knapsack_solver import (
    KnapsackParams,
    MultiPeriodParams,
    check_acoustic_constraints,
    greedy_top_baseline,
    solve_multi_period,
    solve_single_period,
    traditional_multperiod_greedy,
)
from make_synthetic_new_songs import run as build_synthetic
from paths import (
    COMPARISON_CSV,
    CONSTRAINT_CHECK_CSV,
    MULTIPERIOD_SOLUTION_CSV,
    OUTPUT_DIR,
    SENSITIVITY_CSV,
    STRATEGY_REPORT_MD,
    STRATEGY_SUMMARY_CSV,
    resolve_head200_csv,
)
from export_excel import export_strategy_workbook
from score_synthetic_songs import run as score_songs
from song_costs import relative_flop_hop_top


@dataclass
class StrategyConfig:
    csv_path: Optional[str] = None
    base_row: int = BASE_ROW_INDEX
    target: Optional[str] = None
    n_songs: int = N_SYNTHETIC_SONGS
    # 多檔期（預設啟用，含全部限制式）
    run_multperiod: bool = True
    periods: list[str] = field(default_factory=lambda: list(DEFAULT_PERIODS))
    budget_by_period: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BUDGET_BY_PERIOD))
    l_min_by_period: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_L_MIN_BY_PERIOD))
    u_max_by_period: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_U_MAX_BY_PERIOD))
    acoustic_policy: dict = field(default_factory=lambda: dict(DEFAULT_ACOUSTIC_POLICY))
    calibrate_acoustic: bool = True
    acoustic_quantile: float = 0.20
    # 單檔期對照（敏感度用）
    budget: float = sum(DEFAULT_BUDGET_BY_PERIOD.values())
    l_min: int = sum(DEFAULT_L_MIN_BY_PERIOD.values())
    u_max: int = sum(DEFAULT_U_MAX_BY_PERIOD.values())
    auto_feasible: bool = True
    loudness_min: Optional[float] = None
    tempo_min: Optional[float] = None
    budget_sensitivity: tuple[float, ...] = (120.0, 150.0, 180.0, 210.0, 240.0)


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    target: str
    n_valid_target: int
    target_mean: float
    target_std: float
    n_term_features: int
    n_audio_features: int
    has_danceability: bool
    has_energy: bool
    prototype_title: str
    prototype_artist: str
    prototype_hotness: float


def profile_dataset(df: pd.DataFrame, target: str, base_row: int) -> DatasetProfile:
    y = pd.to_numeric(df[target], errors='coerce')
    valid = (y > 0).sum()
    term_cols = [c for c in df.columns if c.startswith('term_')]
    feat_cols = feature_columns(df, target)
    row = df.iloc[base_row]
    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        target=target,
        n_valid_target=int(valid),
        target_mean=float(y[y > 0].mean()) if valid else float('nan'),
        target_std=float(y[y > 0].std()) if valid else float('nan'),
        n_term_features=len(term_cols),
        n_audio_features=len(feat_cols),
        has_danceability='danceability' in df.columns,
        has_energy='energy' in df.columns,
        prototype_title=str(row.get('title', '')),
        prototype_artist=str(row.get('artist_name', '')),
        prototype_hotness=float(pd.to_numeric(row.get(target, np.nan), errors='coerce')),
    )


def classify_flop_hop_top(scores: pd.DataFrame) -> pd.DataFrame:
    """Traditional A&R tiers: 候選池內相對分位（獨立排序，非背包）。"""
    out = scores.copy()
    out['flop_hop_top'] = relative_flop_hop_top(out['y_hat'])
    out['rank_by_yhat'] = out['y_hat'].rank(ascending=False, method='dense').astype(int)
    return out.sort_values('y_hat', ascending=False)


def build_multiperiod_params(cfg: StrategyConfig, scores: pd.DataFrame) -> tuple[MultiPeriodParams, dict, Optional[str]]:
    """組裝多檔期參數並校準聲學門檻 T_jk。"""
    acoustics = cfg.acoustic_policy
    note = None
    if cfg.calibrate_acoustic:
        acoustics = calibrate_acoustic_thresholds(
            scores, policy=cfg.acoustic_policy, quantile=cfg.acoustic_quantile
        )
        note = f'聲學門檻已依候選曲 {cfg.acoustic_quantile:.0%} 分位數校準（確保 ILP 可行）。'
    mparams = MultiPeriodParams(
        periods=list(cfg.periods),
        budget_by_period=dict(cfg.budget_by_period),
        l_min_by_period=dict(cfg.l_min_by_period),
        u_max_by_period=dict(cfg.u_max_by_period),
        acoustic_min_by_period=acoustics,
    )
    return mparams, acoustics, note


def schedule_summary(schedule: pd.DataFrame) -> dict:
    if schedule is None or schedule.empty:
        return {'n_selected': 0, 'total_y_hat': 0.0, 'total_cost': 0.0, 'selected_ids': ''}
    return {
        'n_selected': len(schedule),
        'total_y_hat': float(schedule['y_hat'].sum()),
        'total_cost': float(schedule['cost'].sum()),
        'selected_ids': ','.join(schedule['new_song_id'].tolist()),
    }


def build_comparison_table(
    ilp_df: pd.DataFrame,
    trad_df: pd.DataFrame,
    mparams: MultiPeriodParams,
) -> pd.DataFrame:
    ilp_sum = schedule_summary(ilp_df)
    trad_sum = schedule_summary(trad_df)
    ilp_ac = check_acoustic_constraints(ilp_df, mparams)
    trad_ac = check_acoustic_constraints(trad_df, mparams)
    ilp_pass = bool(ilp_ac['pass'].all()) if len(ilp_ac) else True
    trad_pass = bool(trad_ac['pass'].all()) if len(trad_ac) else False

    rows = [
        {
            '方法': 'ILP 多限制式背包（本模型）',
            '選中首數': ilp_sum['n_selected'],
            'Σŷ': round(ilp_sum['total_y_hat'], 4),
            '總成本': round(ilp_sum['total_cost'], 0),
            '選中 ID': ilp_sum['selected_ids'],
            '聲學門檻全通過': ilp_pass,
            'Δŷ vs 傳統': round(ilp_sum['total_y_hat'] - trad_sum['total_y_hat'], 4),
        },
        {
            '方法': '傳統 A&R（各檔 ŷ 貪婪）',
            '選中首數': trad_sum['n_selected'],
            'Σŷ': round(trad_sum['total_y_hat'], 4),
            '總成本': round(trad_sum['total_cost'], 0),
            '選中 ID': trad_sum['selected_ids'],
            '聲學門檻全通過': trad_pass,
            'Δŷ vs 傳統': 0.0,
        },
    ]
    for k in mparams.periods:
        for label, df in [('ILP', ilp_df), ('傳統', trad_df)]:
            sub = df[df['period'] == k] if df is not None and len(df) else pd.DataFrame()
            s = schedule_summary(sub)
            rows.append({
                '方法': f'{label} · {k}',
                '選中首數': s['n_selected'],
                'Σŷ': round(s['total_y_hat'], 4),
                '總成本': round(s['total_cost'], 0),
                '選中 ID': s['selected_ids'],
                '聲學門檻全通過': '',
                'Δŷ vs 傳統': '',
            })
    return pd.DataFrame(rows)


def ensure_feasible_params(scores: pd.DataFrame, params: KnapsackParams) -> tuple[KnapsackParams, Optional[str]]:
    """若 budget 無法滿足 L_MIN，自動下修 L_MIN 並回傳警告訊息。"""
    min_cost = float(scores['cost'].min())
    max_n = int(params.budget // min_cost) if min_cost > 0 else params.u_max
    if max_n < params.l_min:
        new_l = max(1, max_n)
        msg = (
            f'預算 {params.budget} 不足以發行 {params.l_min} 首（最低成本 {min_cost:.0f}/首），'
            f'已將 L_MIN 調整為 {new_l}。建議提高 --budget 至至少 {params.l_min * min_cost:.0f}。'
        )
        return KnapsackParams(
            budget=params.budget,
            l_min=new_l,
            u_max=min(params.u_max, max_n) if max_n > 0 else params.u_max,
            loudness_min=params.loudness_min,
            tempo_min=params.tempo_min,
        ), msg
    if params.u_max > max_n:
        return KnapsackParams(
            budget=params.budget,
            l_min=params.l_min,
            u_max=max_n,
            loudness_min=params.loudness_min,
            tempo_min=params.tempo_min,
        ), f'U_MAX 已調整為 {max_n}（預算上限）'
    return params, None


def budget_sensitivity(scores: pd.DataFrame, budgets: tuple[float, ...], l_min: int, u_max: int) -> pd.DataFrame:
    rows = []
    for b in budgets:
        params = KnapsackParams(budget=b, l_min=l_min, u_max=u_max)
        try:
            chosen, status = solve_single_period(scores, params)
            sol = scores[scores['new_song_id'].isin(chosen)]
            rows.append({
                'budget': b,
                'status': status,
                'n_selected': len(chosen),
                'total_y_hat': sol['y_hat'].sum() if len(sol) else 0.0,
                'total_cost': sol['cost'].sum() if len(sol) else 0.0,
                'selected_ids': ','.join(chosen),
            })
        except Exception as e:
            rows.append({'budget': b, 'status': str(e), 'n_selected': 0, 'total_y_hat': 0, 'total_cost': 0, 'selected_ids': ''})
    return pd.DataFrame(rows)


def build_strategy_report(
    cfg: StrategyConfig,
    csv_path: Path,
    profile: DatasetProfile,
    ranked: pd.DataFrame,
    chosen: list[str],
    knapsack_params: KnapsackParams,
    baseline: pd.DataFrame,
    sensitivity: pd.DataFrame,
    multperiod_df: Optional[pd.DataFrame] = None,
    feasibility_note: Optional[str] = None,
) -> str:
    sol = ranked[ranked['new_song_id'].isin(chosen)]
    top3_independent = ranked.head(3)

    lines = [
        '# Hit Song Intelligence — 策略分析報告',
        '',
        f'產生時間: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        f'資料來源: `{csv_path}`',
        '',
        '## 1. 資料集診斷',
        '',
        f'| 項目 | 值 |',
        f'|---|---|',
        f'| 列數 × 欄數 | {profile.n_rows} × {profile.n_cols} |',
        f'| 目標變數 | `{profile.target}` |',
        f'| 有效熱度樣本 (>{0}) | {profile.n_valid_target} |',
        f'| 熱度 mean ± std | {profile.target_mean:.4f} ± {profile.target_std:.4f} |',
        f'| 曲風 term_* 欄位數 | {profile.n_term_features} |',
        f'| 模型特徵數 | {profile.n_audio_features} |',
        f'| 原型列 index | {cfg.base_row} |',
        f'| 原型曲名 / 藝人 | {profile.prototype_title} / {profile.prototype_artist} |',
        f'| 原型真實熱度 | {profile.prototype_hotness:.4f} |',
        '',
        f'## 2. 模擬新歌（{cfg.n_songs} 首）',
        '',
        '由 **1 首原型** 調整 loudness / tempo / duration / year / danceability / energy 與音訊微擾生成，',
        '不作為歷史曲庫直接決策（符合期末「假設新歌」設定）。',
        '',
        '## 3. FLOP–HOP–TOP 獨立排序（傳統 A&R）',
        '',
        '依預測 ŷ 分級，**不考慮預算**：',
        '',
        '| 排名 | ID | 情境 | ŷ | 分級 |',
        '|---:|---|---|---:|---|',
    ]
    for _, r in ranked.iterrows():
        lines.append(
            f"| {int(r['rank_by_yhat'])} | {r['new_song_id']} | {r['profile_name']} | {r['y_hat']:.4f} | {r['flop_hop_top']} |"
        )

    lines += [
        '',
        '若只簽 TOP 3（獨立思維）:',
        ', '.join(top3_independent['new_song_id'].tolist()),
        '',
        '## 4. 背包最優決策（全局組合）',
        '',
        f'| 參數 | 值 |',
        f'|---|---|',
        f'| 預算 C | {knapsack_params.budget} |',
        f'| 最少發行 L | {knapsack_params.l_min} |',
        f'| 最多發行 U | {knapsack_params.u_max} |',
        f'| loudness 平均下限 | {knapsack_params.loudness_min or "未啟用"} |',
        f'| tempo 平均下限 | {knapsack_params.tempo_min or "未啟用"} |',
        '',
    ]
    if feasibility_note:
        lines.append(f'> ⚠️ {feasibility_note}')
        lines.append('')
    lines += [
        '### 最終選中曲目',
        '',
        '| ID | 情境 | ŷ | 成本 | tier |',
        '|---|---|---:|---:|---|',
    ]
    for _, r in sol.sort_values('y_hat', ascending=False).iterrows():
        lines.append(
            f"| {r['new_song_id']} | {r['profile_name']} | {r['y_hat']:.4f} | {r['cost']:.0f} | {r['tier']} |"
        )
    lines += [
        '',
        f'**合計** ŷ = {sol["y_hat"].sum():.4f}，成本 = {sol["cost"].sum():.0f}，首數 = {len(sol)}',
        '',
        '### 對照：TOP 貪婪（預算內依 ŷ 由高到低）',
        '',
        ', '.join(baseline['new_song_id'].tolist()) if len(baseline) else '(無)',
        f' — 合計 ŷ = {baseline["y_hat"].sum():.4f}',
        '',
        f'**Δŷ（背包 − 貪婪）** = {sol["y_hat"].sum() - baseline["y_hat"].sum():.4f}',
        '',
        '## 5. 預算敏感度',
        '',
        '| 預算 | 選中首數 | 總 ŷ | 總成本 | 選中 ID |',
        '|---:|---:|---:|---:|---|',
    ]
    for _, r in sensitivity.iterrows():
        lines.append(
            f"| {r['budget']:.0f} | {int(r['n_selected'])} | {r['total_y_hat']:.4f} | {r['total_cost']:.0f} | {r['selected_ids']} |"
        )

    if multperiod_df is not None and len(multperiod_df):
        lines += [
            '',
            '## 6. 多檔期排程（x_ik）',
            '',
            '| 檔期 | ID | 情境 | ŷ |',
            '|---|---|---|---:|',
        ]
        for _, r in multperiod_df.iterrows():
            lines.append(
                f"| {r['period']} | {r['new_song_id']} | {r['profile_name']} | {r['y_hat']:.4f} |"
            )

    lines += [
        '',
        '## 7. 決策建議（簡報用）',
        '',
        '1. **科學輸入**：背包使用模型預測 ŷ，非 MSD 歷史曲庫直接排序。',
        '2. **商業輸入**：成本分級 TOP/HOP/FLOP 與預算 C 請與組員對齊後寫入 `config_columns.py` / `StrategyConfig`。',
        '3. **敘事重點**：若 Δŷ > 0，強調「HOP 組合優於全押 TOP」；若為 0，則強調「在約束下已達可行最優」。',
        '4. **限制聲明**：head200 樣本小，CV R² 僅供參考；決策為輔助排序而非保證爆款。',
        '',
        '---',
        '產出檔：`outputs/strategy_report.md`、`strategy_summary.csv`、`knapsack_solution.csv`',
    ]
    return '\n'.join(lines)


def run_strategy(cfg: StrategyConfig) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = resolve_head200_csv(cfg.csv_path)
    df = load_head200(csv_path)
    target = cfg.target or find_target(df)
    profile = profile_dataset(df, target, cfg.base_row)

    print('=' * 60)
    print('Step 1/5  資料診斷')
    print('=' * 60)
    print(f'  檔案: {csv_path}')
    print(f'  形狀: {profile.n_rows} x {profile.n_cols}')
    print(f'  目標: {target}  有效 n={profile.n_valid_target}  mean={profile.target_mean:.4f}')
    print(f'  原型: [{cfg.base_row}] {profile.prototype_title} — {profile.prototype_artist}')

    print('\n' + '=' * 60)
    print(f'Step 2/5  建立 {cfg.n_songs} 首模擬新歌')
    print('=' * 60)
    build_synthetic(csv_path, cfg.base_row)

    print('\n' + '=' * 60)
    print('Step 3/5  預測熱度 ŷ')
    print('=' * 60)
    scores, train_metrics = score_songs(csv_path, cfg.base_row)
    ranked = classify_flop_hop_top(scores)

    print('\n' + '=' * 60)
    print('Step 4/5  多檔期 ILP + 傳統決策比較')
    print('=' * 60)

    mparams, acoustics_cal, cal_note = build_multiperiod_params(cfg, scores)
    if cal_note:
        print(f'  [校準] {cal_note}')

    multperiod_df, mstatus = solve_multi_period(scores, mparams)
    if mstatus != 'Optimal':
        raise RuntimeError(
            f'多檔期背包不可行: {mstatus}。請放寬 budget_by_period / L/U 或 acoustic_policy。'
        )
    multperiod_df.to_csv(MULTIPERIOD_SOLUTION_CSV, index=False)

    trad_df = traditional_multperiod_greedy(scores, mparams)
    trad_df.to_csv(OUTPUT_DIR / 'traditional_multperiod_solution.csv', index=False)

    comparison = build_comparison_table(multperiod_df, trad_df, mparams)
    comparison.to_csv(COMPARISON_CSV, index=False)

    constraint_check = pd.concat([
        check_acoustic_constraints(multperiod_df, mparams).assign(method='ILP'),
        check_acoustic_constraints(trad_df, mparams).assign(method='傳統'),
    ], ignore_index=True)
    constraint_check.to_csv(CONSTRAINT_CHECK_CSV, index=False)

    chosen = multperiod_df['new_song_id'].tolist()
    ilp_sum = schedule_summary(multperiod_df)
    trad_sum = schedule_summary(trad_df)

    # 單檔期對照（全期總預算）
    kparams = KnapsackParams(
        budget=cfg.budget,
        l_min=cfg.l_min,
        u_max=min(cfg.u_max, len(scores)),
        loudness_min=cfg.loudness_min,
        tempo_min=cfg.tempo_min,
    )
    feas_note = cal_note
    if cfg.auto_feasible:
        kparams, auto_note = ensure_feasible_params(scores, kparams)
        if auto_note:
            feas_note = (feas_note or '') + (' ' if feas_note else '') + auto_note
    single_chosen, status = solve_single_period(scores, kparams)
    baseline = greedy_top_baseline(scores, cfg.budget)
    sensitivity = budget_sensitivity(scores, cfg.budget_sensitivity, cfg.l_min, cfg.u_max)

    full = scores.copy()
    full['selected'] = full['new_song_id'].isin(chosen).astype(int)
    full['flop_hop_top'] = ranked.set_index('new_song_id').loc[full['new_song_id'], 'flop_hop_top'].values
    full['rank_by_yhat'] = ranked.set_index('new_song_id').loc[full['new_song_id'], 'rank_by_yhat'].values
    if 'period' not in full.columns:
        pmap = multperiod_df.set_index('new_song_id')['period'].to_dict() if len(multperiod_df) else {}
        full['period'] = full['new_song_id'].map(pmap)
    full.to_csv(OUTPUT_DIR / 'knapsack_solution.csv', index=False)

    summary = pd.DataFrame([{
        'csv': str(csv_path),
        'base_row': cfg.base_row,
        'n_songs': cfg.n_songs,
        'target': target,
        'total_budget': sum(cfg.budget_by_period.values()),
        'budget_by_period': json.dumps(cfg.budget_by_period, ensure_ascii=False),
        'knapsack_status': mstatus,
        'n_selected_ilp': ilp_sum['n_selected'],
        'selected_ids_ilp': ilp_sum['selected_ids'],
        'total_y_hat_ilp': ilp_sum['total_y_hat'],
        'total_cost_ilp': ilp_sum['total_cost'],
        'n_selected_traditional': trad_sum['n_selected'],
        'total_y_hat_traditional': trad_sum['total_y_hat'],
        'total_cost_traditional': trad_sum['total_cost'],
        'delta_yhat_ilp_vs_traditional': ilp_sum['total_y_hat'] - trad_sum['total_y_hat'],
        'top_greedy_yhat': baseline['y_hat'].sum(),
        'top3_independent': ','.join(ranked.head(3)['new_song_id'].tolist()),
        'acoustic_thresholds': json.dumps(acoustics_cal, ensure_ascii=False),
    }])
    summary.to_csv(STRATEGY_SUMMARY_CSV, index=False)
    sensitivity.to_csv(SENSITIVITY_CSV, index=False)

    report = build_strategy_report(
        cfg, csv_path, profile, ranked, chosen, kparams, baseline, sensitivity,
        multperiod_df, feas_note,
    )
    report += '\n\n## 8. 傳統 vs ILP 比較\n\n'
    report += '```\n' + comparison.to_string(index=False) + '\n```\n'
    STRATEGY_REPORT_MD.write_text(report, encoding='utf-8')

    cv_r2_text = (
        f'5-fold CV R²={train_metrics["cv_r2_mean"]:.4f} ± {train_metrics["cv_r2_std"]:.4f}，'
        f'訓練 n={train_metrics["n_train"]}，特徵數={train_metrics["n_features"]}。'
    )

    excel_path = export_strategy_workbook(
        csv_path=csv_path,
        profile=asdict(profile),
        cfg=asdict(cfg),
        ranked=ranked,
        chosen=chosen,
        kparams={'budget_by_period': cfg.budget_by_period, 'l_min_by_period': cfg.l_min_by_period,
                 'u_max_by_period': cfg.u_max_by_period, 'acoustic': acoustics_cal},
        summary_row=summary.iloc[0].to_dict(),
        baseline=baseline,
        sensitivity=sensitivity,
        full_solution=full,
        multperiod_df=multperiod_df,
        trad_df=trad_df,
        comparison=comparison,
        constraint_check=constraint_check,
        feas_note=feas_note,
        cv_r2_text=cv_r2_text,
    )

    meta = {
        'config': asdict(cfg),
        'profile': asdict(profile),
        'chosen': chosen,
        'status': mstatus,
        'comparison': comparison.to_dict(orient='records'),
    }
    (OUTPUT_DIR / 'strategy_meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )

    print('\n' + '=' * 60)
    print('完成 — 決策摘要')
    print('=' * 60)
    print(comparison.head(2).to_string(index=False))
    print(f'\n  ILP Σŷ={ilp_sum["total_y_hat"]:.4f}  傳統 Σŷ={trad_sum["total_y_hat"]:.4f}  '
          f'Δŷ={ilp_sum["total_y_hat"]-trad_sum["total_y_hat"]:.4f}')
    print(f'  報告: {STRATEGY_REPORT_MD}')
    print(f'  Excel: {excel_path}')

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description='MSD 分析決策策略主程式（20 首 · 多檔期 · 全限制式）')
    ap.add_argument('--csv', type=str, default=None, help='head200 CSV 路徑')
    ap.add_argument('--base-row', type=int, default=BASE_ROW_INDEX)
    ap.add_argument('--budget', type=float, default=None, help='單檔期敏感度用總預算（預設=各檔加總）')
    ap.add_argument('--no-auto-feasible', action='store_true')
    ap.add_argument('--no-calibrate-acoustic', action='store_true', help='不校準聲學門檻')
    ap.add_argument('--single-period-only', action='store_true', help='僅跑單檔期（舊版對照）')
    ap.add_argument('--sensitivity-only', action='store_true')
    args = ap.parse_args()

    total_budget = args.budget if args.budget is not None else sum(DEFAULT_BUDGET_BY_PERIOD.values())
    cfg = StrategyConfig(
        csv_path=args.csv,
        base_row=args.base_row,
        budget=total_budget,
        run_multperiod=not args.single_period_only,
        calibrate_acoustic=not args.no_calibrate_acoustic,
        auto_feasible=not args.no_auto_feasible,
    )

    if args.sensitivity_only:
        from paths import SCORES_CSV
        if not SCORES_CSV.exists():
            print('請先執行 python strategy.py 產生 song_scores.csv', file=sys.stderr)
            sys.exit(1)
        scores = pd.read_csv(SCORES_CSV)
        sens = budget_sensitivity(scores, cfg.budget_sensitivity, cfg.l_min, cfg.u_max)
        sens.to_csv(SENSITIVITY_CSV, index=False)
        print(sens.to_string(index=False))
        return

    try:
        run_strategy(cfg)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
