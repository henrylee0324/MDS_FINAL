"""
Step 12: Add track-level Echo Nest features (extracted from H5 in step 10)
to the existing artist-level pipeline.

The H5 file gave us 11 track-level fields not in our SQLite:
  loudness, tempo, key, key_confidence, mode, mode_confidence,
  time_signature, time_signature_confidence, duration, end_of_fade_in,
  start_of_fade_out

Aggregated per artist into 14 numeric features (mean for continuous,
fraction for binary mode, mode-of-distribution for keys, plus stds).

Compares HistGB (with current best vocab = top-1000 multi-hot) with and
without these additions, on both strict and full configs. Target stays
artist_hotttnesss (this is step C of A->C->B; step B switches target).

Outputs:
  results/12_track_level_features/benchmark_track_level.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / 'data' / 'MSD_with_all_features.db'
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'
HERE    = Path(__file__).resolve().parent
CACHE   = HERE / 'cache'
RESULT  = HERE / 'results' / '12_track_level_features'
RESULT.mkdir(exist_ok=True, parents=True)

NUMERIC_FULL   = ['year_mean','duration_mean','year_known_ratio','n_tracks','n_genres']
NUMERIC_STRICT = ['year_mean','duration_mean','year_known_ratio']

NEW_RAW = ['loudness', 'tempo', 'key', 'key_confidence', 'mode',
           'mode_confidence', 'time_signature',
           'time_signature_confidence', 'end_of_fade_in', 'start_of_fade_out']

def aggregate_track_features(extract):
    """Per-artist aggregation of track-level analysis features."""
    # Some fields are 0/missing for many tracks; mask before stats.
    # loudness in dB is typically -30..0; treat 0 as missing for the rare exact-zeros.
    e = extract.copy()
    e.loc[e['loudness'] == 0, 'loudness'] = np.nan
    # tempo of 0 means unknown
    e.loc[e['tempo'] == 0, 'tempo'] = np.nan
    # key/mode 0 are valid (C / minor) so don't mask those
    # time_signature 0 means unknown
    e.loc[e['time_signature'] == 0, 'time_signature'] = np.nan

    g = e.groupby('artist_id', sort=False)
    agg = pd.DataFrame({
        'tl_loudness_mean':       g['loudness'].mean(),
        'tl_loudness_std':        g['loudness'].std(),
        'tl_tempo_mean':          g['tempo'].mean(),
        'tl_tempo_std':           g['tempo'].std(),
        'tl_key_confidence_mean': g['key_confidence'].mean(),
        'tl_mode_fraction':       g['mode'].mean(),
        'tl_mode_confidence_mean':g['mode_confidence'].mean(),
        'tl_time_sig_mean':       g['time_signature'].mean(),
        'tl_time_sig_confidence_mean': g['time_signature_confidence'].mean(),
        'tl_fade_in_mean':        g['end_of_fade_in'].mean(),
        'tl_fade_out_start_mean': g['start_of_fade_out'].mean(),
        'tl_fade_in_std':         g['end_of_fade_in'].std(),
    }).reset_index()
    return agg

def main():
    t0 = time.time()
    audio = pd.read_pickle(CACHE / 'artist_audio_agg.pkl')
    extra = pd.read_pickle(CACHE / 'artist_extra.pkl')
    extract = pd.read_pickle(EXTRACT)
    print(f'Loaded extract: {extract.shape}   ({time.time()-t0:.1f}s)')

    # Restrict track-level extract to the pipeline artist universe
    valid_artists = set(audio['artist_id'])
    e = extract[extract['artist_id'].isin(valid_artists)]
    print(f'Tracks restricted to pipeline artists: {len(e):,} of {len(extract):,}')

    tl = aggregate_track_features(e)
    print(f'Per-artist track-level aggregation: {tl.shape}')
    tl_cols = [c for c in tl.columns if c.startswith('tl_')]

    # Audio + numeric setup (same as before)
    audio_feats = [c for c in audio.columns if c not in ('artist_id','hotness','n_tracks')]
    mom_prune    = [c for c in audio_feats if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_feats if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_feats if c not in set(mom_prune)|set(marsyas_mfcc)]
    clean = audio.dropna(subset=audio_feats).reset_index(drop=True)
    Xa = clean[audio_keep].values.astype(np.float64)
    audio_keep = [k for k, kp in zip(audio_keep, Xa.std(axis=0)>0) if kp]

    # Re-fetch raw term for per-artist top-N tag construction
    print('Fetching term column for top-1000 tag multi-hot...')
    with sqlite3.connect(DB) as conn:
        terms = pd.read_sql("""
            SELECT artist_id, MAX(term) AS term
            FROM merged_partition1
            WHERE CAST(artist_hotttnesss AS REAL) > 0
            GROUP BY artist_id
        """, conn)

    df = (clean.merge(extra, on='artist_id')
                .merge(terms, on='artist_id')
                .merge(tl,    on='artist_id', how='left'))
    print(f'Joined: {df.shape}   missing tl rows after join: {df[tl_cols[0]].isna().sum()}')

    df['term_list'] = df['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])
    df['term_set'] = df['term_list'].apply(set)
    counter = Counter()
    for lst in df['term_list']:
        counter.update(lst)
    top1000_tags = [t for t, _ in counter.most_common(1000)]
    M_top1000 = np.zeros((len(df), 1000), dtype=np.float32)
    for j, tag in enumerate(top1000_tags):
        M_top1000[:, j] = df['term_set'].apply(lambda s: tag in s).values
    print(f'Built top-1000 multi-hot: {M_top1000.shape}')

    y = df['hotness'].values
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def cv_r2(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.03, max_depth=8,
            l2_regularization=0.1, min_samples_leaf=20, random_state=42)
        scores = cross_val_score(hgb, X, y, cv=cv, scoring='r2', n_jobs=-1)
        return scores.mean(), scores.std()

    def stack(*blocks):
        return np.hstack([b.astype(np.float32) for b in blocks])

    Xa_arr     = df[audio_keep].values.astype(np.float32)
    Xnum_full  = df[NUMERIC_FULL].values.astype(np.float32)
    Xnum_strict= df[NUMERIC_STRICT].values.astype(np.float32)
    Xtl        = df[tl_cols].values.astype(np.float32)

    rows = []
    print(f'\n=== HistGB benchmarks: with vs without track-level features ({time.time()-t0:.1f}s) ===')
    for cfg_name, num_block in [('strict', Xnum_strict), ('full', Xnum_full)]:
        # baseline: top-1000 tags, no tl
        X = stack(Xa_arr, num_block, M_top1000)
        m, s = cv_r2(X)
        rows.append({'config':cfg_name, 'add_tl':False, 'n_feat':X.shape[1],
                     'cv_r2':m, 'cv_std':s})
        print(f'  {cfg_name:>6s}, top-1000, NO tl     : R^2 = {m:.4f} +/- {s:.4f}'
              f'   ({X.shape[1]} feats)   ({time.time()-t0:.1f}s)')

        # +track-level
        X = stack(Xa_arr, num_block, M_top1000, Xtl)
        m, s = cv_r2(X)
        rows.append({'config':cfg_name, 'add_tl':True, 'n_feat':X.shape[1],
                     'cv_r2':m, 'cv_std':s})
        print(f'  {cfg_name:>6s}, top-1000, +tl ({len(tl_cols)})  : R^2 = {m:.4f} +/- {s:.4f}'
              f'   ({X.shape[1]} feats)   ({time.time()-t0:.1f}s)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_track_level.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_track_level.csv"} ===')
    print(bench.to_string(index=False))

    print('\n=== Reference (step 09) ===')
    print('  HistGB strict + top-1000:  R^2 = 0.4278')
    print('  HistGB full   + top-1000:  R^2 = 0.4618')
    print('  CatBoost full + text(7538):R^2 = 0.4698')

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
