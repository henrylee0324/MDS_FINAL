"""
Step A: Build 10 synthetic 'new songs' from ONE prototype row in head200 CSV.

Usage:
  python make_synthetic_new_songs.py
  python make_synthetic_new_songs.py --csv ../data/head200_MSD_with_all_features_categorical_encoded.csv
  python make_synthetic_new_songs.py --base-row 5
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from audio_sentinel import fix_audio_sentinels
from config_columns import ACOUSTIC_FOR_CONSTRAINTS, BASE_ROW_INDEX, PROFILES
from io_utils import load_head200
from paths import MOCK_CSV, OUTPUT_DIR, SYNTHETIC_CSV, resolve_head200_csv


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def perturb_row(row: pd.Series, prof: dict, rng: np.random.Generator) -> pd.Series:
    out = row.copy()
    for c in _numeric_cols(out.to_frame().T):
        if c in ('year', 'duration', 'loudness', 'tempo'):
            continue
        v = out[c]
        if pd.notna(v) and np.isfinite(v):
            out[c] = float(v) * float(rng.uniform(0.97, 1.03))

    if 'loudness' in out.index and pd.notna(out['loudness']):
        out['loudness'] = float(out['loudness']) + prof['loudness_delta']
    if 'tempo' in out.index and pd.notna(out['tempo']) and float(out['tempo']) > 0:
        out['tempo'] = float(out['tempo']) * prof['tempo_scale']
    if 'duration' in out.index and pd.notna(out['duration']) and float(out['duration']) > 0:
        out['duration'] = float(out['duration']) * prof['duration_scale']
    if 'year' in out.index and pd.notna(out['year']):
        y = int(float(out['year']))
        if y > 0:
            out['year'] = max(1950, min(2025, y + prof['year_delta']))
    for key, col in (('danceability_delta', 'danceability'), ('energy_delta', 'energy')):
        if col in out.index and key in prof:
            v = float(out[col]) if pd.notna(out[col]) else 0.0
            out[col] = float(np.clip(v + prof[key], 0, 1))
    return out


def run(csv_path, base_row: int) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_head200(csv_path)
    print(f'Loaded {csv_path.name}: {df.shape[0]} rows x {df.shape[1]} cols')

    found = [c for c in ACOUSTIC_FOR_CONSTRAINTS if c in df.columns]
    print(f'Acoustic columns: {found or "(none)"}')

    if base_row >= len(df):
        raise ValueError(f'base_row={base_row} out of range (max {len(df)-1})')

    base = fix_audio_sentinels(df.iloc[base_row].copy())
    print(f'Prototype row: {base_row} (audio sentinels normalized)')

    rng = np.random.default_rng(42)
    rows = []
    for i, prof in enumerate(PROFILES, start=1):
        row = perturb_row(base, prof, rng)
        row['new_song_id'] = f'NEW_SONG_{i:02d}'
        row['profile_name'] = prof['profile_name']
        row['base_row_index'] = base_row
        if 'track_id' in row.index:
            row['source_track_id_masked'] = 'REDACTED_PROTOTYPE'
        rows.append(row)

    syn = pd.DataFrame(rows)
    syn.to_csv(SYNTHETIC_CSV, index=False)
    print(f'Wrote {len(syn)} songs -> {SYNTHETIC_CSV}')
    show = ['new_song_id', 'profile_name'] + [c for c in found if c in syn.columns]
    print(syn[show].to_string(index=False))
    return syn


def main() -> None:
    from pathlib import Path

    ap = argparse.ArgumentParser(description='Create 10 synthetic new songs from one prototype row.')
    ap.add_argument('--csv', type=str, default=None, help='Path to head200 CSV')
    ap.add_argument('--base-row', type=int, default=BASE_ROW_INDEX)
    ap.add_argument('--use-mock', action='store_true', help='Use generated mock CSV if real file missing')
    args = ap.parse_args()

    if args.use_mock:
        if not MOCK_CSV.exists():
            print('Mock CSV missing. Run: python create_mock_head200.py', file=sys.stderr)
            sys.exit(1)
        csv_path = MOCK_CSV
        print(f'Using mock data: {csv_path}')
    else:
        try:
            csv_path = resolve_head200_csv(args.csv)
        except FileNotFoundError as e:
            print(f'{e}\nPlace CSV in MDS/ or MDS_FINAL-main/data/, or use --use-mock', file=sys.stderr)
            sys.exit(1)

    run(csv_path, args.base_row)


if __name__ == '__main__':
    main()
