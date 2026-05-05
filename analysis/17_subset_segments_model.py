"""
Step 17: Does adding segment-level features (step 16) improve song-level
prediction R^2 on the MSD 10K subset?

Restricts the song-level pipeline (step 13 setup) to the 10K subset's
track_ids and runs HistGB twice on the same data:

  baseline   = our existing 152 audio + 10 tl + year + duration
               + top-1000 genre multi-hot (+ optional 3 leak features)
  enhanced   = baseline + 101 segment-derived features from step 16

5-fold GroupKFold by artist_id.

Sample size after filtering ~6-9K (depending on song_hotttnesss > 0
filter overlap with our pipeline). At this size CV noise is ~+/-0.02,
so report should focus on DELTA R^2 (baseline vs enhanced) which uses
the same 10K cohort.

Outputs:
  results/17_subset_segments_model/benchmark_subset.csv
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
SEGFEATS= ROOT / 'data' / 'subset_segment_features.pkl'
HERE    = Path(__file__).resolve().parent
RESULT  = HERE / 'results' / '17_subset_segments_model'
RESULT.mkdir(exist_ok=True, parents=True)

METADATA_COLS = {'track_id','title','song_id','release','artist_id','artist_mbid',
                 'artist_name','duration','artist_familiarity','artist_hotttnesss',
                 'year','track_7digitalid','shs_perf','shs_work','term','similar'}
TL_COLS = ['loudness','tempo','key','key_confidence','mode','mode_confidence',
           'time_signature','time_signature_confidence',
           'end_of_fade_in','start_of_fade_out']
NUMERIC_FULL   = ['n_tracks_a','n_genres_a','year_known_ratio_a']

def main():
    t0 = time.time()
    seg = pd.read_pickle(SEGFEATS)
    print(f'Subset segment features: {seg.shape}   ({time.time()-t0:.1f}s)')
    seg_cols = [c for c in seg.columns if c != 'track_id']
    subset_ids = set(seg['track_id'])

    extract = pd.read_pickle(EXTRACT)
    valid = extract[(extract['song_hotttnesss'] > 0) & extract['track_id'].isin(subset_ids)].copy()
    print(f'Subset & song_hot>0: {len(valid):,}')

    # Pull audio + term per song for these track_ids
    cols = pd.read_sql('PRAGMA table_info(merged_partition1)',
                       sqlite3.connect(DB))['name'].tolist()
    audio_cols = [c for c in cols if c not in METADATA_COLS]
    audio_keep = [c for c in audio_cols
                  if not c.startswith('Method_of_Moments_')
                  and 'Mem20_MFCC' not in c]

    print('Querying SQLite for these track_ids...')
    valid_track_ids = set(valid['track_id'])
    sel = ['track_id','artist_id','term'] + [f'"{c}"' for c in audio_keep]
    chunks = []
    with sqlite3.connect(DB) as conn:
        for ch in pd.read_sql(
            f"SELECT {', '.join(sel)} FROM merged_partition1 "
            "WHERE CAST(artist_hotttnesss AS REAL) > 0",
            conn, chunksize=100_000):
            chunks.append(ch[ch['track_id'].isin(valid_track_ids)])
    audio_df = pd.concat(chunks, ignore_index=True)
    print(f'Audio matched: {audio_df.shape}')

    # Drop zero-var audio
    Xc = audio_df[audio_keep].values.astype(np.float64)
    keep_idx = (~np.isnan(Xc).all(axis=0)) & (np.nanstd(Xc, axis=0) > 0)
    audio_keep = [k for k, kp in zip(audio_keep, keep_idx) if kp]
    audio_df = audio_df.dropna(subset=audio_keep, how='all').reset_index(drop=True)
    print(f'Audio cols after zero-var: {len(audio_keep)}, rows: {len(audio_df)}')

    merge_cols = ['track_id','song_hotttnesss','year','duration'] + TL_COLS
    df = audio_df.merge(valid[merge_cols], on='track_id').merge(seg, on='track_id')
    print(f'Joined (audio + extract + seg): {df.shape}')

    # Per-artist agg for full
    g = df.groupby('artist_id')
    artist_agg = pd.DataFrame({
        'n_tracks_a':         g.size(),
        'year_known_ratio_a': g['year'].apply(lambda s: (s > 0).mean()),
    }).reset_index()
    df = df.merge(artist_agg, on='artist_id')
    df['term_list'] = df['term'].fillna('').apply(
        lambda s: [t.strip() for t in s.split(',') if t.strip()])
    df['term_set']  = df['term_list'].apply(set)
    df['n_genres_a'] = df['term_list'].apply(len)

    counter = Counter()
    for lst in df['term_list']:
        counter.update(lst)
    top1000_tags = [t for t, _ in counter.most_common(1000)]
    print(f'Genre vocab in subset: {len(counter)} unique; top-1000 used')
    M = np.zeros((len(df), 1000), dtype=np.float32)
    for j, tag in enumerate(top1000_tags):
        M[:, j] = df['term_set'].apply(lambda s: tag in s).values

    df.loc[df['loudness'] == 0, 'loudness'] = np.nan
    df.loc[df['tempo'] == 0, 'tempo'] = np.nan
    df.loc[df['time_signature'] == 0, 'time_signature'] = np.nan
    year_known = (df['year'] > 0).astype(np.float32).values.reshape(-1,1)
    df.loc[df['year'] == 0, 'year'] = np.nan

    Xa  = df[audio_keep].values.astype(np.float32)
    Xtl = df[TL_COLS].values.astype(np.float32)
    Xyd = np.column_stack([df['year'].values, df['duration'].values, year_known.ravel()]).astype(np.float32)
    Xex = df[NUMERIC_FULL].values.astype(np.float32)
    Xseg= df[seg_cols].values.astype(np.float32)
    y   = df['song_hotttnesss'].values.astype(np.float32)
    groups = df['artist_id'].values

    print(f'\nDataset finalized: n={len(df):,}  artists={len(np.unique(groups)):,}')
    print(f'  audio={Xa.shape[1]}  tl={Xtl.shape[1]}  yd=3  extras=3  '
          f'genre_top1000={M.shape[1]}  seg={Xseg.shape[1]}')

    cv = GroupKFold(n_splits=5)
    def cv_r2(X):
        hgb = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=8,
            l2_regularization=0.1, min_samples_leaf=20, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        scores = cross_val_score(hgb, X, y, cv=cv, groups=groups,
                                  scoring='r2', n_jobs=-1)
        return scores.mean(), scores.std()

    rows = []
    print(f'\n=== Configurations (5-fold GroupKFold) ===')

    # strict baseline
    X = np.hstack([Xa, Xtl, Xyd, M])
    print(f'\n  strict baseline (audio+tl+yd+top1000): X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'strict_baseline', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # strict + segment features
    X = np.hstack([Xa, Xtl, Xyd, M, Xseg])
    print(f'\n  strict + segment ({Xseg.shape[1]}): X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'strict_plus_seg', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # full baseline
    X = np.hstack([Xa, Xtl, Xyd, Xex, M])
    print(f'\n  full baseline (+ 3 leak): X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'full_baseline', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # full + segment
    X = np.hstack([Xa, Xtl, Xyd, Xex, M, Xseg])
    print(f'\n  full + segment: X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'full_plus_seg', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # segment-only (no audio, no genre)
    X = np.hstack([Xseg, Xyd])
    print(f'\n  segment-only + yd (no audio, no genre): X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'segment_only', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # baseline 152 audio only (no seg, no genre, no tl) for sanity
    X = Xa
    print(f'\n  audio_only (152): X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'audio_only_152', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    # audio + seg only (test: do segments add to audio block alone?)
    X = np.hstack([Xa, Xseg])
    print(f'\n  audio + seg: X={X.shape}')
    m, s = cv_r2(X); rows.append({'config':'audio_plus_seg', 'n_feat':X.shape[1],
                                   'cv_r2':m, 'cv_std':s})
    print(f'  -> R^2 = {m:.4f} +/- {s:.4f}')

    bench = pd.DataFrame(rows)
    bench.to_csv(RESULT / 'benchmark_subset.csv', index=False)
    print(f'\n=== Saved -> {RESULT/"benchmark_subset.csv"} ===')
    print(bench.to_string(index=False))

    # Compute deltas
    base = {r['config']: r['cv_r2'] for r in rows}
    print('\n=== Delta from adding segment features (the headline) ===')
    print(f'  strict + seg vs baseline:  '
          f'{base["strict_plus_seg"]-base["strict_baseline"]:+.4f}')
    print(f'  full   + seg vs baseline:  '
          f'{base["full_plus_seg"]-base["full_baseline"]:+.4f}')
    print(f'  audio  + seg vs audio_only:'
          f'{base["audio_plus_seg"]-base["audio_only_152"]:+.4f}')

    print(f'\nDone in {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
