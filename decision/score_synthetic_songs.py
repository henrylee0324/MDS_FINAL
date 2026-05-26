"""
Step B: Train HistGB on head200 (excluding prototype row), predict y_hat for 10 synthetic songs.

Usage:
  python score_synthetic_songs.py
  python score_synthetic_songs.py --csv path/to/head200.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

from config_columns import BASE_ROW_INDEX
from song_costs import assign_profile_costs
from io_utils import feature_columns, find_target, load_head200
from paths import MOCK_CSV, MODEL_PKL, OUTPUT_DIR, SCORES_CSV, SYNTHETIC_CSV, resolve_head200_csv


def run(csv_path: Path, base_row: int) -> tuple[pd.DataFrame, dict]:
    if not SYNTHETIC_CSV.exists():
        raise FileNotFoundError(f'Run make_synthetic_new_songs.py first. Missing {SYNTHETIC_CSV}')

    raw = load_head200(csv_path)
    syn = pd.read_csv(SYNTHETIC_CSV)
    target = find_target(raw)
    print(f'Target: {target}')

    train_df = raw.drop(index=base_row, errors='ignore').reset_index(drop=True)
    train_df[target] = pd.to_numeric(train_df[target], errors='coerce')
    train_df = train_df[train_df[target] > 0].reset_index(drop=True)

    feat_cols = feature_columns(train_df, target)
    for c in feat_cols:
        if c not in syn.columns:
            syn[c] = np.nan

    X_train = train_df[feat_cols].astype(np.float32)
    y_train = train_df[target].astype(np.float32).values
    X_syn = syn[feat_cols].astype(np.float32)

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('hgb', HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    n_splits = min(5, max(2, len(train_df) // 15))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=1)
    print(f'Train n={len(train_df)}  features={len(feat_cols)}  CV R2={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

    y_hat = np.clip(model.predict(X_syn), 0, None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PKL, 'wb') as f:
        pickle.dump({'model': model, 'feature_cols': feat_cols, 'target': target}, f)

    out = syn[['new_song_id', 'profile_name']].copy()
    out['y_hat'] = y_hat
    if target in syn.columns:
        out['y_true_holdout'] = pd.to_numeric(syn[target], errors='coerce')
    for c in ('loudness', 'tempo', 'duration', 'year', 'danceability', 'energy'):
        if c in syn.columns:
            out[c] = syn[c].values

    out = assign_profile_costs(out)

    out.to_csv(SCORES_CSV, index=False)
    print(f'Saved model -> {MODEL_PKL}')
    print(f'Saved scores -> {SCORES_CSV}')
    print(out.sort_values('y_hat', ascending=False).to_string(index=False))
    metrics = {
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'n_train': len(train_df),
        'n_features': len(feat_cols),
        'target': target,
    }
    return out, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None)
    ap.add_argument('--base-row', type=int, default=BASE_ROW_INDEX)
    ap.add_argument('--use-mock', action='store_true')
    args = ap.parse_args()

    try:
        csv_path = resolve_head200_csv(args.csv)
    except FileNotFoundError:
        if MOCK_CSV.exists():
            csv_path = MOCK_CSV
        else:
            print('Missing head200 CSV. Place file in MDS/ or MDS_FINAL-main/data/', file=sys.stderr)
            sys.exit(1)

    run(csv_path, args.base_row)


if __name__ == '__main__':
    main()
