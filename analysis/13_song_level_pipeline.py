"""
Step 13: Song-level pipeline with song_hotttnesss as target.

Steps 01-12 worked at the artist level (one row per artist; all features
averaged). With song_hotttnesss now available we can model at the SONG
level (one row per track), keeping every track-level feature ungrouped.

Critical change: 5-fold cross-validation uses GroupKFold(artist_id)
instead of plain KFold. Same-artist tracks are correlated; if random
splits put some of them in train and others in test, R^2 is inflated
("you've seen this artist before" leakage).

Feature blocks at song level:
  audio_per_track (152)  - same 152 columns from merged_partition1, but
                           NOT averaged - one row per track
  tl_per_track (10)      - loudness, tempo, key, key_conf, mode, mode_conf,
                           time_sig, time_sig_conf, end_of_fade_in,
                           start_of_fade_out (per-track from H5)
  year, duration (2)     - per-track, not artist mean
  genre top-1000 (1000)  - per-artist multi-hot, repeated for each track
                           of that artist
  numeric_full (3 extra) - n_tracks, n_genres, year_known_ratio at artist
                           level (still leakage at song level)

Configurations (5-fold GroupKFold by artist_id):
  strict_song  =  audio_per_track + tl + year + duration + top-1000 tags
  full_song    =  strict_song + n_tracks + n_genres + year_known_ratio

Outputs:
  results/13_song_level_pipeline/benchmark_song_level.csv
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import sqlite3, time
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

ROOT    = Path(__file__).resolve().parent.parent
DB      = ROOT / 'data' / 'MSD_with_all_features.db'
EXTRACT = ROOT / 'data' / 'msd_summary_extract.pkl'
HERE    = Path(__file__).resolve().parent
RESULT  = HERE / 'results' / '13_song_level_pipeline'
RESULT.mkdir(exist_ok=True, parents=True)

METADATA_COLS = {'track_id','title','song_id','release','artist_id','artist_mbid',
                 'artist_name','duration','artist_familiarity','artist_hotttnesss',
                 'year','track_7digitalid','shs_perf','shs_work','term','similar'}

TL_COLS = ['loudness','tempo','key','key_confidence','mode','mode_confidence',
           'time_signature','time_signature_confidence',
           'end_of_fade_in','start_of_fade_out']

def main():
    t0 = time.time()
    extract = pd.read_pickle(EXTRACT)
    print(f'Loaded extract: {extract.shape}   ({time.time()-t0:.1f}s)')

    valid = extract[extract['song_hotttnesss'] > 0].copy()
    print(f'Tracks with song_hotttnesss > 0: {len(valid):,}')

    # Drop rows where any tl feature is NaN/zero-zero (we'll keep loudness=NaN later
    # because HistGB handles it natively; only filter on track_id presence)
    valid_track_ids = set(valid['track_id'])

    # Pull per-track audio features + artist_id + term from SQLite
    print('Querying merged_partition1 for matching tracks...')
    cols_check = pd.read_sql('PRAGMA table_info(merged_partition1)', sqlite3.connect(DB))['name'].tolist()
    audio_cols = [c for c in cols_check if c not in METADATA_COLS]
    mom_prune    = [c for c in audio_cols if c.startswith('Method_of_Moments_')]
    marsyas_mfcc = [c for c in audio_cols if 'Mem20_MFCC' in c]
    audio_keep   = [c for c in audio_cols if c not in set(mom_prune)|set(marsyas_mfcc)]

    # Build SQL select for needed cols (use chunks to avoid 1M-row OOM)
    sel = ['track_id','artist_id','term'] + [f'"{c}"' for c in audio_keep]
    sql = f"""
        SELECT {', '.join(sel)}
        FROM merged_partition1
        WHERE CAST(artist_hotttnesss AS REAL) > 0
    """
    chunks = []
    with sqlite3.connect(DB) as conn:
        for chunk in pd.read_sql(sql, conn, chunksize=100_000):
            chunks.append(chunk[chunk['track_id'].isin(valid_track_ids)])
    audio_df = pd.concat(chunks, ignore_index=True)
    print(f'Audio (per-track, restricted to song_hot>0): {audio_df.shape}   ({time.time()-t0:.1f}s)')

    # Drop zero-variance audio cols (relative to this filtered subset)
    Xa_check = audio_df[audio_keep].values.astype(np.float64)
    keep_idx = (~np.isnan(Xa_check).all(axis=0)) & (np.nanstd(Xa_check, axis=0) > 0)
    audio_keep = [k for k, kp in zip(audio_keep, keep_idx) if kp]
    print(f'Audio cols after zero-var drop: {len(audio_keep)}')

    # Drop tracks where ALL audio is NaN
    audio_df = audio_df.dropna(subset=audio_keep, how='all').reset_index(drop=True)
    print(f'After dropping all-NaN audio rows: {audio_df.shape}')

    # Merge with extract for song_hot, year, duration, tl features
    merge_cols = ['track_id','song_hotttnesss','year','duration'] + TL_COLS
    df = audio_df.merge(valid[merge_cols], on='track_id', how='inner')
    print(f'Merged (audio + extract): {df.shape}   ({time.time()-t0:.1f}s)')

    # Build per-artist features and merge
    g = df.groupby('artist_id')
    artist_feat = pd.DataFrame({
        'n_tracks_a':      g.size(),
        'year_known_ratio_a': g['year'].apply(lambda s: (s > 0).mean()),
    }).reset_index()

    df['term_list'] = df['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])
    artist_genre = (df.groupby('artist_id')['term_list']
                      .first().reset_index())
    artist_genre['n_genres_a'] = artist_genre['term_list'].apply(len)
    df = df.merge(artist_feat, on='artist_id').merge(
        artist_genre[['artist_id','n_genres_a']], on='artist_id')

    print(f'After per-artist join: {df.shape}')

    # Build top-1000 multi-hot
    counter = Counter()
    for lst in df['term_list']:
        counter.update(lst)
    top1000_tags = [t for t, _ in counter.most_common(1000)]
    print(f'Genre vocab: {len(counter):,} unique; using top-1000')
    M = np.zeros((len(df), 1000), dtype=np.float32)
    df['term_set'] = df['term_list'].apply(set)
    for j, tag in enumerate(top1000_tags):
        M[:, j] = df['term_set'].apply(lambda s: tag in s).values
    print(f'Built top-1000 multi-hot: {M.shape}   ({time.time()-t0:.1f}s)')

    # Replace tl 0/sentinels with NaN (HistGB handles NaN natively)
    df.loc[df['loudness'] == 0, 'loudness'] = np.nan
    df.loc[df['tempo'] == 0, 'tempo'] = np.nan
    df.loc[df['time_signature'] == 0, 'time_signature'] = np.nan
    # Year=0 -> NaN as a per-track signal
    year_known = (df['year'] > 0).astype(np.float32)
    df.loc[df['year'] == 0, 'year'] = np.nan

    y = df['song_hotttnesss'].values.astype(np.float32)
    groups = df['artist_id'].values

    # Build feature blocks
    X_audio  = df[audio_keep].values.astype(np.float32)
    X_tl     = df[TL_COLS].values.astype(np.float32)
    X_yd     = np.column_stack([df['year'].values, df['duration'].values,
                                year_known.values]).astype(np.float32)  # 3 cols
    X_extra  = df[['n_tracks_a','n_genres_a','year_known_ratio_a']].values.astype(np.float32)

    print(f'\nDataset finalized: n={len(df):,}  groups={df["artist_id"].nunique():,}')

    cv = GroupKFold(n_splits=5)
    def cv_r2(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=8,
            l2_regularization=0.1, min_samples_leaf=50, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        scores = cross_val_score(hgb, X, y, cv=cv, groups=groups,
                                  scoring='r2', n_jobs=-1)
        return scores.mean(), scores.std()

    rows = []

    # strict_song: audio + tl + per-track year/duration + top-1000 tags
    print(f'\n=== Running configs (each ~10-15 min) ===')
    X = np.hstack([X_audio, X_tl, X_yd, M])
    print(f'  strict_song: X.shape={X.shape}   ({time.time()-t0:.1f}s)')
    m, s = cv_r2(X)
    rows.append({'config':'strict_song', 'n_feat':X.shape[1], 'cv_r2':m, 'cv_std':s})
    print(f'  strict_song : R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    # full_song: + 3 artist-level extras (still leakage flavoured)
    X = np.hstack([X_audio, X_tl, X_yd, X_extra, M])
    print(f'  full_song:   X.shape={X.shape}   ({time.time()-t0:.1f}s)')
    m, s = cv_r2(X)
    rows.append({'config':'full_song', 'n_feat':X.shape[1], 'cv_r2':m, 'cv_std':s})
    print(f'  full_song   : R^2 = {m:.4f} +/- {s:.4f}   ({time.time()-t0:.1f}s)')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_song_level.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_song_level.csv"} ===')
    print(bench.to_string(index=False))

    print('\n=== Reference (artist-level, KFold, target=artist_hotttnesss) ===')
    print('  HistGB strict + top-1000 (step 09):           R^2 = 0.4278')
    print('  HistGB strict + top-1000 + tl (step 12):      R^2 = 0.4330')
    print('  HistGB full   + top-1000 (step 09):           R^2 = 0.4618')
    print('  HistGB full   + top-1000 + tl (step 12):      R^2 = 0.4635')
    print('  CatBoost full + text(7538) (step 08):          R^2 = 0.4698')

    print(f'\nDone in {time.time()-t0:.1f}s.')

if __name__ == '__main__':
    main()
