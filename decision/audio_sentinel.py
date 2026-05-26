"""Normalize MSD sentinel values before perturbation / constraints."""
from __future__ import annotations

import pandas as pd

DEFAULT_TEMPO_BPM = 120.0
DEFAULT_DURATION = 200_000.0  # ms-like scale when column is raw MSD duration


def fix_audio_sentinels(row: pd.Series) -> pd.Series:
    out = row.copy()
    if 'tempo' in out.index:
        t = float(out['tempo']) if pd.notna(out['tempo']) else 0.0
        if t <= 0:
            out['tempo'] = DEFAULT_TEMPO_BPM
    if 'duration' in out.index:
        d = float(out['duration']) if pd.notna(out['duration']) else 0.0
        if d <= 0 or d < 10:  # tiny values = normalized / missing
            out['duration'] = DEFAULT_DURATION
    if 'loudness' in out.index and pd.isna(out['loudness']):
        out['loudness'] = -10.0
    if 'year' in out.index:
        y = float(out['year']) if pd.notna(out['year']) else 0.0
        if y <= 0:
            out['year'] = 2010
    return out
