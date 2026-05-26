"""Shared I/O and feature-matrix helpers."""
from __future__ import annotations

import pandas as pd

from config_columns import ID_COLS, TARGET_CANDIDATES


def find_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f'No target column in CSV. Add name to config_columns.TARGET_CANDIDATES. '
        f'Tried: {TARGET_CANDIDATES}'
    )


def feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    leak_cols = {'song_hotttnesss', 'artist_hotttnesss', 'artist_familiarity'}
    if target in leak_cols:
        leak_cols.discard(target)
    skip = set(ID_COLS) | leak_cols | {target, 'y_true_holdout', 'y_hat', 'cost', 'tier', 'selected', 'period'}
    feats = []
    for c in df.columns:
        if c in skip or c.endswith('_true'):
            continue
        if c in ('source_track_id_masked', 'term'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    return feats


def load_head200(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)
